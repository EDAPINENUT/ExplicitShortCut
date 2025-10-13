import math
import torch
import torch.nn.functional as F
import numpy as np
import torch.func
from utils.scheduler import CosineFlowScheduler, LinearFlowScheduler
from utils import append_dims

class SCMCosineLoss:
    """Simplifying, Stablizing, Scaling Consistency Model (SCM) loss.

    Supports cosine and linear flow schedules, optional variational adaptive
    weighting, and classifier-free guidance via label dropout in the caller.
    """
    def __init__(
        self,
        path_type="cosine",
        time_sampler="logit_normal",
        # New parameters
        time_mu=-1,
        time_sigma=1.4,
        grad_warmup_steps=10000,
        loss_type="l2",
        adaptive_p=1.0,
        label_dropout_prob=0.1,
        variational_adaptive_weight=False
    ):
        """Initialize loss configuration and flow scheduler.

        Args:
            path_type (str): "cosine" or "linear" schedule for the flow.
            time_mu (float): Mean for log-normal time sampling in latent time.
            time_sigma (float): Std for log-normal time sampling in latent time.
            grad_warmup_steps (int): Steps to linearly warm up gradient term.
            loss_type (str): One of {"l2", "l1", "adaptive"}.
            adaptive_p (float): Power used in adaptive weighting.
            label_dropout_prob (float): Probability of label dropout for CFG (handled
                in caller via `y` masking to `num_classes`).
            variational_adaptive_weight (bool): If True, expects model to output
                `(pred, log_var)` and applies NLL-style weighting.
        """
        self.path_type = path_type
        
        # Noise sampling config
        self.time_sampler = time_sampler
        self.time_mu = time_mu
        self.time_sigma = time_sigma

        # Time sampling config
        if path_type == "cosine":
            self.sigma_data = 0.5
            self.flow_scheduler = CosineFlowScheduler(sigma_data=self.sigma_data)
            self.grad_weight = lambda t: torch.cos(t * math.pi / 2)
        else:
            raise NotImplementedError(f"Path type {path_type} not implemented")

        self.grad_warmup_steps = grad_warmup_steps
        self.loss_type = loss_type
        self.adaptive_p = adaptive_p
        self.label_dropout_prob = label_dropout_prob
        self.variational_adaptive_weight = variational_adaptive_weight

    def sample_time_steps(self, batch_size, device):
        """Sample times `(r, s, t)` according to the configured sampler.

        For cosine: samples `t` via arctan transform; for linear: via sigmoid on
        log-normal draws. In both cases, `s = t` and `r = 0`.

        Args:
            batch_size (int): Number of time samples to generate.
            device (torch.device): Device for returned tensors.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: `r`, `s`, `t` of shape
            `(B,)`, with `s == t` and `r == 0`.
        """
        # Step1: Sample one time points
        if self.time_sampler != "logit_normal":
            raise ValueError(f"Unknown time sampler: {self.time_sampler}")
        t_normal = torch.randn(batch_size, device=device)
        t_lognormal = (t_normal * self.time_sigma + self.time_mu).exp() 
        if self.path_type == "cosine":
            t = torch.arctan(t_lognormal / self.sigma_data) / math.pi * 2
        else:
            raise NotImplementedError(f"Path type {self.path_type} not implemented")
        s = t
        r = torch.zeros_like(t)
        return r, s, t

    def interpolant(self, t):
        """Compute interpolant scalars and their derivatives at time `t`.

        Args:
            t (torch.Tensor): Time tensor broadcastable to `(B, 1, 1, 1)`.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            `(alpha_t, sigma_t, d_alpha_t, d_sigma_t)`.
        """
        alpha_t = self.flow_scheduler.alpha(t) # α(t) = cos(πt/2)
        sigma_t = self.flow_scheduler.sigma(t) # σ(t) = sin(πt/2)
        d_alpha_t = self.flow_scheduler.d_alpha(t) # dα(t)/dt = -π/2 * sin(πt/2)
        d_sigma_t = self.flow_scheduler.d_sigma(t) # dσ(t)/dt = π/2 * cos(πt/2)
        return alpha_t, sigma_t, d_alpha_t, d_sigma_t
    
    def _v_pred(self, model, z_t, t, r, y):
        """Call model to predict velocity (optionally with log variance).

        Args:
            model (Callable): Model `F_theta(c_in(t) * z_t, r, t, y, ...)`.
            z_t (torch.Tensor): Interpolated state `(B, C, H, W)`.
            t (torch.Tensor): Times `(B,)`.
            r (torch.Tensor): Start times `(B,)`.
            y (torch.Tensor): Labels `(B,)` or similar.

        Returns:
            torch.Tensor | Tuple[torch.Tensor, torch.Tensor]: `v_pred` or
            `(v_pred, log_var)` if `variational_adaptive_weight` is True.
        """
        t_ = append_dims(t, z_t.ndim)
        c_in = self.flow_scheduler.c_in(t_)
        if self.variational_adaptive_weight:
            return model(c_in * z_t, r, t, y, return_logvar=True)
        return model(c_in * z_t, r, t, y) 
    
    def __call__(self, model, model_tgt, images, kwargs=None):
        """Compute SCM loss for a batch.

        Args:
            model (Callable): Student model returning `v_pred` (and optional `log_var`).
            model_tgt (Callable): Target/EMA model used inside `_tgt_v`.
            images (torch.Tensor): Input batch `(B, C, H, W)`.
            kwargs (Optional[dict]): Must include `global_step` and `y`. If label
                dropout is enabled, `y` is cloned and masked to `num_classes`.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: `(loss, loss_mid)` where `loss` is
            per-sample and `loss_mid` is a reference (pre-weight or per-sample metric).
        """
        batch_size = images.shape[0]
        device = images.device
        global_step = kwargs["global_step"]
        y = kwargs["y"]
        
        if kwargs.get('y') is not None and self.label_dropout_prob > 0:
            y = kwargs['y'].clone()  
            batch_size = y.shape[0]
            num_classes = model_tgt.num_classes
            dropout_mask = torch.rand(batch_size, device=y.device) < self.label_dropout_prob
            
            y[dropout_mask] = num_classes

        r, s, t = self.sample_time_steps(batch_size, device)
        t_ = append_dims(t, images.ndim)
        r_ = append_dims(r, images.ndim)
        s_ = append_dims(s, images.ndim)

        noises = torch.randn_like(images) * self.sigma_data
        alpha_t, sigma_t, d_alpha_t, d_sigma_t = self.interpolant(t_)
        z_t = alpha_t * images + sigma_t * noises
        v_t = d_alpha_t * images + d_sigma_t * noises
        
        v_pred = self._v_pred(model, z_t, t, r, y)
        if self.variational_adaptive_weight:
            v_pred, log_var = v_pred
        else:
            log_var = torch.zeros(batch_size, device=device)
        v_tgt = self._tgt_v(model_tgt, z_t, v_t, t, s, r, y, global_step)
        
        loss, loss_mid = self.loss_v(v_pred, v_tgt)
        loss = 1 / torch.exp(log_var) * loss + log_var
        return loss, loss_mid
    
    def loss_v(self, v_pred, v_tgt):
        """Compute loss between predicted and target velocities.

        Args:
            v_pred (torch.Tensor): Predicted velocity `(B, C, H, W)`.
            v_tgt (torch.Tensor): Target velocity `(B, C, H, W)`.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Per-sample loss and reference metric.
        """
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

    def _tgt_v(self, model_tgt, z_t, v_t, t, s, r, y, global_step):
        """Compute gradient-guided target `v_tgt` with warmup and normalization.

        Implements Eq.(8)-style target formation using JVP with flow-scheduler
        coefficients, warmup on the second-order term, and tangent normalization.

        Args:
            model_tgt (Callable): EMA/teacher model `F_theta`.
            z_t (torch.Tensor): Interpolated state `(B, C, H, W)`.
            v_t (torch.Tensor): Interpolant velocity `(B, C, H, W)`.
            t (torch.Tensor): Times `(B,)`.
            s (torch.Tensor): Same as `t` `(B,)`.
            r (torch.Tensor): Start times `(B,)`.
            y (torch.Tensor): Labels `(B,)`.
            global_step (int): Global step for warmup scheduling.

        Returns:
            torch.Tensor: Target velocity `(B, C, H, W)`.
        """
        assert (s == t).all()
        dim = z_t.ndim
        t_ = append_dims(t, dim)
        r_ = append_dims(r, dim)
        c_in_t = self.flow_scheduler.c_in(t_)
        beta_bar_t = self.flow_scheduler.beta_bar(t_, r_) # -sin(πt/2)
        wg_t = self.grad_weight(t_) # cos(πt/2)
        coeff_z_t = beta_bar_t * wg_t * v_t * c_in_t
        coeff_t = beta_bar_t * wg_t

        def fn_current(z, cur_r, cur_t):
            return model_tgt(z, cur_r, cur_t, y)
        
        v_tgt, dv_dt_tgt = torch.func.jvp(
            fn_current,
            (c_in_t * z_t, r, t),  
            (coeff_z_t, torch.zeros_like(r), coeff_t.flatten()) 
        )

        d_beta_bar_t = self.flow_scheduler.d_beta_bar_dt(t_, r_) # -cos(πt/2)
        alpha_bar_t = self.flow_scheduler.alpha_bar(t_, r_) # cos(πt/2)
        g_ = wg_t * (d_beta_bar_t * v_tgt + alpha_bar_t * v_t) # cos(πt/2)*(-cos(πt/2) * c_out_t * F_tgt + cos(πt/2) * v_t)

        d_alpha_bar_t = self.flow_scheduler.d_alpha_bar_dt(t_, r_) # -sin(πt/2) * π/2
        def _warmup(g, dv_dt_tgt, z_t, global_step):
            r = min(1.0, global_step / self.grad_warmup_steps)
            # Note that F_theta_grad_tgt is already multiplied by beta_bar_t * wg_t from the tangents. Doing it early helps with stability.
            second_term = r * (d_alpha_bar_t * wg_t * z_t + dv_dt_tgt) # sin(πt/2) * π/2
            return g + second_term
        g_warmup = _warmup(g_, dv_dt_tgt, z_t, global_step)

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
        
        return g + v_tgt  # Note that it is F_theta - F_theta_minus in Eq(8) in SCM, not c_out * F_theta_minus