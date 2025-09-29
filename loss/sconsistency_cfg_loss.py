import math
import torch
import torch.nn.functional as F
import numpy as np
import torch.func
from functools import partial
from scheduler import CosineFlowScheduler, LinearFlowScheduler
from utils import append_dims
import os

class SConsistencyLoss:
    def __init__(
        self,
        path_type="cosine",
        # New parameters
        time_mu=-1,
        time_sigma=1.4,
        grad_warmup_steps=10000,
        loss_type="l2",
        adaptive_p=1.0,
        label_dropout_prob=0.1,
        cfg_min_t=0.0,
        cfg_max_t=0.8,
        cfg_omega=1.0,
        cfg_kappa=0.0,
    ):
        self.path_type = path_type
        
        # Noise sampling config
        self.time_mu = time_mu
        self.time_sigma = time_sigma

        # Time sampling config
        if path_type == "cosine":
            self.sigma_data = 0.5
            self.flow_scheduler = CosineFlowScheduler(sigma_data=self.sigma_data)
            self.grad_weight = lambda t: torch.cos(t * math.pi / 2)
        elif path_type == "linear":
            self.sigma_data = 1.0
            self.flow_scheduler = LinearFlowScheduler()
            self.grad_weight = lambda t: torch.ones_like(t)
        else:
            raise NotImplementedError(f"Path type {path_type} not implemented")

        self.grad_warmup_steps = grad_warmup_steps
        self.loss_type = loss_type
        self.adaptive_p = adaptive_p
        self.label_dropout_prob = label_dropout_prob
        self.cfg_min_t = cfg_min_t
        self.cfg_max_t = cfg_max_t
        self.cfg_omega = cfg_omega
        self.cfg_kappa = cfg_kappa
        
    def sample_time_steps(self, batch_size, device):
        """Sample time steps (r, t) according to the configured sampler"""
        # Step1: Sample one time points
        t_normal = torch.randn(batch_size, device=device)
        t_lognormal = (t_normal * self.time_sigma + self.time_mu).exp() 
        if self.path_type == "cosine":
            t = torch.arctan(t_lognormal / self.sigma_data) / math.pi * 2
        elif self.path_type == "linear":
            t = torch.sigmoid(t_lognormal.log())
        else:
            raise NotImplementedError(f"Path type {self.path_type} not implemented")
        s = t
        r = torch.zeros_like(t)
        return r, s, t

    def interpolant(self, t):
        """Define interpolation function"""
        alpha_t = self.flow_scheduler.alpha(t) # α(t) = cos(πt/2)
        sigma_t = self.flow_scheduler.sigma(t) # σ(t) = sin(πt/2)
        d_alpha_t = self.flow_scheduler.d_alpha(t) # dα(t)/dt = -π/2 * sin(πt/2)
        d_sigma_t = self.flow_scheduler.d_sigma(t) # dσ(t)/dt = π/2 * cos(πt/2)
        return alpha_t, sigma_t, d_alpha_t, d_sigma_t
    
    def _v_pred(self, model, z_t, t, r, y):
        t_ = append_dims(t, z_t.ndim)
        c_in = self.flow_scheduler.c_in(t_)
        return model(c_in * z_t, r, t, y) 
    
    def __call__(self, model, model_tgt, images, kwargs=None):
        """
        Compute sCM loss function (unconditional)
        """
        batch_size = images.shape[0]
        device = images.device
        global_step = kwargs["global_step"]
        y = kwargs["y"]

        unconditional_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        if kwargs.get('y') is not None and self.label_dropout_prob > 0:
            y = kwargs['y'].clone()  
            batch_size = y.shape[0]
            num_classes = model_tgt.num_classes
            dropout_mask = torch.rand(batch_size, device=y.device) < self.label_dropout_prob
            
            y[dropout_mask] = num_classes
            kwargs['y'] = y
            unconditional_mask = dropout_mask

        r, s, t = self.sample_time_steps(batch_size, device)
        t_ = append_dims(t, images.ndim)
        r_ = append_dims(r, images.ndim)
        s_ = append_dims(s, images.ndim)

        noises = torch.randn_like(images) * self.sigma_data
        alpha_t, sigma_t, d_alpha_t, d_sigma_t = self.interpolant(t_)
        z_t = alpha_t * images + sigma_t * noises
        v_t = d_alpha_t * images + d_sigma_t * noises
        
        v_pred = self._v_pred(model, z_t, t, r, y)
        v_tgt = self._tgt_v(model_tgt, z_t, v_t, t, s, r, y, global_step, unconditional_mask)
        
        loss, loss_mid = self.loss_v(v_pred, v_tgt)
        return loss, loss_mid
    
    def loss_v(self, v_pred, v_tgt):
        v_tgt = v_tgt.detach()
        if self.loss_type == "l2":
            loss =  F.mse_loss(v_pred, v_tgt, reduction="none").mean(dim=(1, 2, 3))
            loss_ref = loss
        elif self.loss_type == "l1":
            loss = F.l1_loss(v_pred, v_tgt, reduction="none").mean(dim=(1, 2, 3))
            loss_ref = loss
        elif self.loss_type == "adaptive":
            error = v_pred - v_tgt.detach()
            loss_mid = torch.sum((error**2).reshape(error.shape[0],-1), dim=-1)
            weights = 1.0 / (loss_mid.detach() ** 2 + 1e-3).pow(self.adaptive_p)
            loss = weights * loss_mid ** 2    
            loss_ref = torch.mean((error**2))
        else:
            raise NotImplementedError(f"Loss type {self.loss_type} not implemented")
        return loss, loss_ref

    def _tgt_v(self, model_tgt, z_t, v_t, t, s, r, y, global_step, unconditional_mask):
        assert (s == t).all()
        dim = z_t.ndim
        t_ = append_dims(t, dim)
        r_ = append_dims(r, dim)
        c_in_t = self.flow_scheduler.c_in(t_)
        beta_bar_t = self.flow_scheduler.beta_bar(t_, r_) # -sin(πt/2)
        wg_t = self.grad_weight(t_) # cos(πt/2)


        batch_size = z_t.shape[0]
        v_target = torch.zeros_like(v_t)    
        dvdt_target = torch.zeros_like(v_t)
        
        # Check if CFG should be applied (exclude unconditional samples)
        cfg_time_mask = (t >= self.cfg_min_t) & (t <= self.cfg_max_t) & (~unconditional_mask)
        
        cfg_indices = torch.where(cfg_time_mask)[0]
        no_cfg_indices = torch.where(~cfg_time_mask)[0]
        if len(cfg_indices) > 0:
            cfg_z_t = z_t[cfg_indices]
            cfg_v_t = v_t[cfg_indices]
            cfg_r = r[cfg_indices]
            cfg_t = t[cfg_indices]
            cfg_y = y[cfg_indices]
            cfg_beta_bar_t = beta_bar_t[cfg_indices]
            cfg_wg_t = wg_t[cfg_indices]
            cfg_c_in_t = c_in_t[cfg_indices]
            num_classes = model_tgt.num_classes
            cfg_z_t_batch = torch.cat([cfg_z_t, cfg_z_t], dim=0)
            cfg_t_batch = torch.cat([cfg_t, cfg_t], dim=0)
            cfg_t_end_batch = torch.cat([cfg_t, cfg_t], dim=0)
            cfg_y_batch = torch.cat([cfg_y, torch.full_like(cfg_y, num_classes)], dim=0)
            
            with torch.no_grad():
                cfg_combined_u_at_t = model_tgt(cfg_z_t_batch, cfg_t_batch, cfg_t_end_batch, y=cfg_y_batch)
                cfg_v_cond_at_t, cfg_v_uncond_at_t = torch.chunk(cfg_combined_u_at_t, 2, dim=0)
                cfg_v_tilde = (self.cfg_omega * cfg_v_t + 
                            self.cfg_kappa * cfg_v_cond_at_t + 
                            (1 - self.cfg_omega - self.cfg_kappa) * cfg_v_uncond_at_t)
            
            def fn_current_cfg(z, cur_r, cur_t):
                return model_tgt(z, cur_r, cur_t, y=cfg_y)
            
            primals = (cfg_z_t, cfg_r, cfg_t)
            cfg_coeff_z_t = cfg_beta_bar_t * cfg_wg_t * cfg_c_in_t * cfg_v_tilde
            cfg_coeff_t = cfg_beta_bar_t * cfg_wg_t
            tangents = (
                cfg_coeff_z_t, 
                torch.zeros_like(cfg_r), 
                cfg_coeff_t.flatten()
            )
            _, cfg_dudt = torch.func.jvp(fn_current_cfg, primals, tangents)
            cfg_v_target = cfg_v_tilde
            cfg_dvdt_target = cfg_dudt
            v_target[cfg_indices] = cfg_v_target
            dvdt_target[cfg_indices] = cfg_dvdt_target
            
        if len(no_cfg_indices) > 0:
            no_cfg_z_t = z_t[no_cfg_indices]
            no_cfg_v_t = v_t[no_cfg_indices]
            no_cfg_r = r[no_cfg_indices]
            no_cfg_t = t[no_cfg_indices]
            no_cfg_y = y[no_cfg_indices]
            no_cfg_beta_bar_t = beta_bar_t[no_cfg_indices]
            no_cfg_wg_t = wg_t[no_cfg_indices]
            no_cfg_c_in_t = c_in_t[no_cfg_indices]

            def fn_current(z, cur_r, cur_t):
                return model_tgt(z, cur_r, cur_t, no_cfg_y)
            
            primals = (no_cfg_z_t, no_cfg_r, no_cfg_t)
            no_cfg_coeff_z_t = no_cfg_beta_bar_t * no_cfg_wg_t * no_cfg_c_in_t * no_cfg_v_t
            no_cfg_coeff_t = no_cfg_beta_bar_t * no_cfg_wg_t
            tangents = (
                no_cfg_coeff_z_t, 
                torch.zeros_like(no_cfg_r), 
                no_cfg_coeff_t.flatten()
            )
            _, no_cfg_dudt = torch.func.jvp(fn_current, primals, tangents)

            no_cfg_v_target = no_cfg_v_t
            no_cfg_dvdt_target = no_cfg_dudt
            v_target[no_cfg_indices] = no_cfg_v_target
            dvdt_target[no_cfg_indices] = no_cfg_dvdt_target

        d_beta_bar_t = self.flow_scheduler.d_beta_bar_dt(t_, r_) # -cos(πt/2)
        alpha_bar_t = self.flow_scheduler.alpha_bar(t_, r_) # cos(πt/2)
        g_ = wg_t * (d_beta_bar_t * v_target + alpha_bar_t * v_t) # cos(πt/2)*(-cos(πt/2) * c_out_t * F_tgt + cos(πt/2) * v_t)

        d_alpha_bar_t = self.flow_scheduler.d_alpha_bar_dt(t_, r_) # -sin(πt/2) * π/2
        def _warmup(g, dv_dt_tgt, z_t, global_step):
            r = min(1.0, global_step / self.grad_warmup_steps)
            # Note that F_theta_grad_tgt is already multiplied by beta_bar_t * wg_t from the tangents. Doing it early helps with stability.
            second_term = r * (d_alpha_bar_t * wg_t * z_t + dv_dt_tgt) # sin(πt/2) * π/2
            return g + second_term
        g_warmup = _warmup(g_, dvdt_target, z_t, global_step)

        def _tangent_normalization(g):
            """Tangent normalization"""
            g_norm = torch.linalg.vector_norm(g, dim=(1, 2, 3), keepdim=True)
            # Multiplying by sqrt(numel(g_norm) / numel(g)) ensures that the norm is invariant to the spatial dimensions.
            g_norm = g_norm * np.sqrt(g_norm.numel() / g.numel()) 
            # 0.1 is the constant c, can be modified but 0.1 was used in the paper
            g = g / (g_norm + 0.1) 
            # g = torch.clamp(g, min=-1, max=1)
            return g
        
        g = _tangent_normalization(g_warmup)
        
        return g + v_target  # Note that it is F_theta - F_theta_minus in Eq(8) in SCM, not c_out * F_theta_minus