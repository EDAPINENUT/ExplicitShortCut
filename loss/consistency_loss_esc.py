import torch
import numpy as np
import torch.func
from functools import partial
from utils.scheduler import CosineFlowScheduler, LinearFlowScheduler
from utils.solver import ddim_solver_condv, ddim_solver_condx0
from utils import append_dims
import math
from torch import Tensor

def lognormal_timestep_distribution(
    num_samples: int,
    sigmas: Tensor,
    mean: float = -1.1,
    std: float = 2.0,
) -> Tensor:
    """Draws timesteps from a lognormal distribution.

    Parameters
    ----------
    num_samples : int
        Number of samples to draw.
    sigmas : Tensor
        Standard deviations of the noise.
    mean : float, default=-1.1
        Mean of the lognormal distribution.
    std : float, default=2.0
        Standard deviation of the lognormal distribution.

    Returns
    -------
    Tensor
        Timesteps drawn from the lognormal distribution.

    References
    ----------
    [1] [Improved Techniques For Consistency Training](https://arxiv.org/pdf/2310.14189.pdf)
    """
    pdf = torch.erf((torch.log(sigmas[1:]) - mean) / (std * math.sqrt(2))) - torch.erf(
        (torch.log(sigmas[:-1]) - mean) / (std * math.sqrt(2))
    )
    pdf = pdf / pdf.sum()

    timesteps = torch.multinomial(pdf, num_samples, replacement=True)

    return timesteps


class ConsistencyLoss:
    def __init__(
        self,
        path_type="cosine",
        # New parameters
        time_sampler="progressive",
        loss_type="l2",
        label_dropout_prob=0.1,
        adaptive_p=1.0,
        cfg_omega=1.0,
        cfg_min_t=0.0,
        cfg_max_t=1.0,
        rho=7.0,
        sigma_max=80.0,
        sigma_min=0.002,
        time_mu=-0.4,
        time_sigma=1.0,
    ):
        self.path_type = path_type
        
        # Time sampling config
        self.time_sampler = time_sampler
        self.sigma_data = 1.0
        if path_type == "cosine":
            self.flow_scheduler = CosineFlowScheduler(sigma_data=self.sigma_data)
        elif path_type == "linear":
            self.flow_scheduler = LinearFlowScheduler()
        else:
            raise NotImplementedError(f"Path type {path_type} not implemented")
        self.loss_type = loss_type
        self.label_dropout_prob = label_dropout_prob

        self.adaptive_p = adaptive_p
        
        # CFG config
        self.cfg_omega = cfg_omega
        self.cfg_min_t = cfg_min_t
        self.cfg_max_t = cfg_max_t
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.rho = rho
        self.time_mu = time_mu
        self.time_sigma = time_sigma

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
        if self.time_sampler == "progressive" and self.path_type == "cosine":
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

            t = torch.arctan(t) / (math.pi / 2)
            s = torch.arctan(s) / (math.pi / 2)
            r = torch.zeros_like(t)
        elif self.time_sampler == "logit_normal" and self.path_type == "linear":
            indices = torch.arange(
                0, scales, device=device
            ) / max(scales - 1, 1)
            discrete_timespace = self.sigma_max ** (1 / self.rho) + indices / (scales - 1) * (
                self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho)
            )
            discrete_timespace = discrete_timespace**self.rho 
            t_sample = lognormal_timestep_distribution(
                batch_size, discrete_timespace, self.time_mu, self.time_sigma
            )
            normed_timespace = discrete_timespace / self.sigma_max
            t = normed_timespace[t_sample].log()
            s = normed_timespace[t_sample+ 1].log()
            t = torch.sigmoid(t)
            s = torch.sigmoid(s)
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

        noises = torch.randn_like(images) * self.flow_scheduler.sigma_0
        alpha_t, sigma_t, d_alpha_t, d_sigma_t = self.interpolant(t_)
        z_t = alpha_t * images + sigma_t * noises
        v_t = d_alpha_t * images + d_sigma_t * noises
        
        alpha_bar = self.flow_scheduler.alpha_bar 
        beta_bar = self.flow_scheduler.beta_bar
        v_pred = self._v_pred(model, z_t, t, r, y)
        z_pred = ddim_solver_condv(z_t, v_pred, t, r, alpha_bar, beta_bar)
        z_tgt = self._tgt_u(model_tgt, z_t, images, noises, t, s, r, y)

        loss_u, loss_u_ref = self.loss_u(z_pred, z_tgt)
        return loss_u, loss_u_ref
    
    def _v_pred(self, model, z_t, t, r, y):
        t_ = append_dims(t, z_t.ndim)
        c_in, c_out = self.flow_scheduler.c_in(t_), self.flow_scheduler.c_out(t_)
        return model(c_in * z_t, r, t, y) * c_out
    
    def _tgt_v(self, model_tgt, z, v, t, r):
        """
        Compute the target v for distillation
        """
        return v
    
    @torch.no_grad()
    def _tgt_u(self, model_tgt, z, x0, noises, t, s, r, y):
        s_ = append_dims(s, z.ndim)
        alpha_s, sigma_s, _, _ = self.interpolant(s_)
        z_s = alpha_s * x0 + sigma_s * noises
        v_s = self._v_pred(model_tgt, z_s, s, r, y)
        alpha_bar = self.flow_scheduler.alpha_bar 
        beta_bar = self.flow_scheduler.beta_bar
        z_tgt = ddim_solver_condv(z_s, v_s, s, r, alpha_bar, beta_bar)
        return z_tgt

    def loss_u(self, u_pred, u_target):
        """
        Compute loss for velocity u
        """
                # Detach the target to prevent gradient flow        
        error = u_pred - u_target.detach()
        # Apply adaptive loss based on configuration
        if self.loss_type == "adaptive":
            loss_mid = torch.sum((error**2).reshape(error.shape[0],-1), dim=-1)
            weights = 1.0 / (loss_mid.detach() ** 2 + 1e-3).pow(self.adaptive_p)
            loss = weights * loss_mid ** 2          
        elif self.loss_type == "l2":
            loss = torch.mean((error**2).reshape(error.shape[0],-1), dim=-1)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        loss_mean_ref = torch.mean((error**2))

        return loss, loss_mean_ref