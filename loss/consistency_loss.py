import torch
import numpy as np
import torch.func
from functools import partial
from utils.scheduler import EDMFlowScheduler
from utils.solver import ddim_solver_condv
from utils import append_dims

class ConsistencyLoss:
    def __init__(
            self,
            path_type="edm",
            # New parameters
            time_sampler="progressive",  
            sigma_min=0.002,
            sigma_max=80,
            loss_type="adaptive",
            adaptive_p=1.0,
            cfg_omega=1.0,
        ):
        self.path_type = path_type
        
        # Time sampling config
        self.time_sampler = time_sampler
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = 0.5
        if path_type == "edm":
            self.flow_scheduler = EDMFlowScheduler(
                sigma_min=self.sigma_min,
                sigma_max=self.sigma_max,
                sigma_data=self.sigma_data,
            )
        else:
            raise NotImplementedError(f"Path type {path_type} not implemented")
        self.loss_type = loss_type
        self.rho = 7.0
        self.adaptive_p = adaptive_p
        self.cfg_omega = cfg_omega

    def interpolant(self, t):
        """Define interpolation function"""
        alpha_t = self.flow_scheduler.alpha(t)
        sigma_t = self.flow_scheduler.sigma(t)
        d_alpha_t = self.flow_scheduler.d_alpha(t)
        d_sigma_t = self.flow_scheduler.d_sigma(t)
        return alpha_t, sigma_t, d_alpha_t, d_sigma_t
    
    def sample_time_steps(self, batch_size, device, scales):
        """Sample time steps (r, t) according to the configured sampler"""
        # Step1: Sample two time points
        if self.time_sampler == "progressive":
            indices = torch.randint(
                0, scales - 1, (batch_size,), device=device
            )
            t = self.sigma_max ** (1 / self.rho) + indices / (scales - 1) * (
                self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho)
            )
            t = t**self.rho

            s = self.sigma_max ** (1 / self.rho) + (indices + 1) / (scales - 1) * (
                self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho)
            )
            s = s**self.rho

            t = t / self.sigma_max
            s = s / self.sigma_max
            r = torch.zeros_like(t)
        else:
            raise NotImplementedError(f"Time sampler {self.time_sampler} not implemented")

        return r, s, t

    def __call__(self, model, model_tgt, images, kwargs=None):
        """
        Compute MeanFlow loss function (unconditional)
        """
        batch_size = images.shape[0]
        device = images.device
        scales = kwargs["scales"]
        y = kwargs["y"]
        # Sample time steps
        r, s, t = self.sample_time_steps(batch_size, device, scales)
        t_ = append_dims(t, images.ndim)
        r_ = append_dims(r, images.ndim)
        s_ = append_dims(s, images.ndim)

        noises = torch.randn_like(images)
        alpha_t, sigma_t, d_alpha_t, d_sigma_t = self.interpolant(t_)
        z_t = alpha_t * images + sigma_t * noises
        v_t = d_alpha_t * images + d_sigma_t * noises

        alpha_bar = self.flow_scheduler.alpha_bar 
        beta_bar = self.flow_scheduler.beta_bar
        v_pred = self._v_pred(model, z_t, t, r, y)
        z_pred = ddim_solver_condv(z_t, v_pred, t, r, alpha_bar, beta_bar)

        z_tgt = self._tgt_u(model_tgt, z_t, images, noises, t, s, r, y)
        loss_u, error = self.loss_u(z_pred, z_tgt)
        return loss_u, error
    
    def _v_pred(self, model, z_t, t, r, y):
        t_ = append_dims(t, z_t.ndim)
        c_in, c_out = self.flow_scheduler.c_in(t_), self.flow_scheduler.c_out(t_)
        if self.cfg_omega > 1.0:
            z_t = torch.cat([z_t, z_t], dim=0)
            t = torch.cat([t, t], dim=0)
            r = torch.cat([r, r], dim=0)
            y = torch.cat([y, torch.zeros_like(y)], dim=0)
            c_in = torch.cat([c_in, c_in], dim=0)
            c_out = torch.cat([c_out, c_out], dim=0)
            model_output = model(c_in * z_t, r, t, y) * c_out
            v_uncond, v_cond = model_output.chunk(2, dim=0)
            return v_uncond + self.cfg_omega * (v_cond - v_uncond)
        else:
            return model(c_in * z_t, r, t, y) * c_out

    def _denoise_fn(self, model, xt, t, r):
        c_skip, c_out, c_in = [
            append_dims(x, xt.ndim) for x in self.get_scalings(t * self.sigma_max)
        ]
        model_output = model(c_in * xt, r, t)
        denoised = c_out * model_output + c_skip * xt
        return denoised
    
    def get_scalings(self, sigma):
        c_skip = self.sigma_data**2 / (
            (sigma - self.sigma_min) ** 2 + self.sigma_data**2
        )
        c_out = (
            (sigma - self.sigma_min)
            * self.sigma_data
            / (sigma**2 + self.sigma_data**2) ** 0.5
        )
        c_in = 1 / (sigma**2 + self.sigma_data**2) ** 0.5
        return c_skip, c_out, c_in

    def _tgt_v(self, model_tgt, z, v, t, r):
        """
        Compute the target v for distillation
        """
        return v
    
    @torch.no_grad()
    def _tgt_u(self, model_tgt, z_t, x0, noises, t, s, r, y):
        s_ = append_dims(s, z_t.ndim)
        z_s = x0 + s_ * noises * self.sigma_max # theoretically the same as euler_solver_condx0(z_t, x0, t, s, self.sigma_max)
        v_s = self._v_pred(model_tgt, z_s, s, r, y)
        alpha_bar = self.flow_scheduler.alpha_bar 
        beta_bar = self.flow_scheduler.beta_bar
        z_tgt = ddim_solver_condv(z_s, v_s, s, r, alpha_bar, beta_bar)
        return z_tgt

    def loss_u(self, u_pred, u_tgt, weights=1.0):
        error = u_pred - u_tgt.detach()
        loss_mid = torch.sum((error**2).reshape(error.shape[0],-1), dim=-1)
        # Apply adaptive weighting based on configuration
        if self.loss_type == "adaptive":
            weights = 1.0 / (loss_mid.detach() ** 2 + 1e-3).pow(self.adaptive_p)
            loss = weights * loss_mid ** 2          
        else:
            loss = loss_mid
        loss_mean_ref = torch.mean((error**2))
        return loss, loss_mean_ref