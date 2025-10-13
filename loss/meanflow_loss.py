
'''
adapted from https://github.com/zhuyu-cs/MeanFlow/blob/main/loss.py
'''

import torch
from torch.func import jvp
from utils.scheduler import LinearFlowScheduler

class MeanFlowLoss:
    """MeanFlow training loss with optional classifier-free guidance (CFG).

    This class computes the target velocity and loss for MeanFlow-based models.
    It supports configurable time sampling strategies, adaptive loss weighting,
    label dropout for classifier-free guidance, and CFG mixing controls.
    """
    def __init__(
        self,
        path_type="linear",
        loss_type="l2",
        # New parameters
        time_sampler="logit_normal",  # Time sampling strategy: "uniform" or "logit_normal"
        time_mu=-0.4,                 # Mean parameter for logit_normal distribution
        time_sigma=1.0,               # Std parameter for logit_normal distribution
        ratio_r_not_equal_t=0.75,     # Ratio of samples where r≠t
        adaptive_p=1.0,               # Power param for adaptive loss
        label_dropout_prob=0.1,       # Drop out label
        # CFG related params
        cfg_omega=1.0,                # CFG omega param, default 1.0 means no CFG
        cfg_kappa=0.0,                # CFG kappa param for mixing class-cond and uncond u
        cfg_min_t=0.0,                # Minium CFG trigger time 
        cfg_max_t=0.8,                # Maximum CFG trigger time
    ):
        """Initialize MeanFlow loss configuration.

        Args:
            path_type (str): Interpolation path type. Currently supports "linear".
            loss_type (str): Loss type. "l2" for mean squared error, or "adaptive" for
                adaptive weighting based on per-sample error magnitude.
            time_sampler (str): Time sampling strategy: "uniform" or "logit_normal".
            time_mu (float): Mean parameter for the logit-normal time sampler.
            time_sigma (float): Std parameter for the logit-normal time sampler.
            ratio_r_not_equal_t (float): Fraction in (0,1] specifying the probability
                that sampled times satisfy r < t; remaining fraction sets r = t.
            adaptive_p (float): Power for adaptive loss weighting when `loss_type` is
                "adaptive".
            label_dropout_prob (float): Probability to drop labels (set to class `num_classes`)
                to create unconditional samples for CFG.
            cfg_omega (float): CFG mixing weight for the instantaneous velocity term.
            cfg_kappa (float): CFG mixing weight for the conditional target velocity.
            cfg_min_t (float): Minimum time to enable CFG mixing.
            cfg_max_t (float): Maximum time to enable CFG mixing.
        """
        self.loss_type = loss_type
        self.path_type = path_type
        
        # Time sampling config
        self.time_sampler = time_sampler
        self.time_mu = time_mu
        self.time_sigma = time_sigma
        self.ratio_r_not_equal_t = ratio_r_not_equal_t
        self.label_dropout_prob = label_dropout_prob
        # Adaptive weight config
        self.adaptive_p = adaptive_p
        
        # CFG config
        self.cfg_omega = cfg_omega
        self.cfg_kappa = cfg_kappa
        self.cfg_min_t = cfg_min_t
        self.cfg_max_t = cfg_max_t
        if path_type == "linear":
            self.flow_scheduler = LinearFlowScheduler()
        

    def interpolant(self, t):
        """Compute interpolation scalars and their derivatives at time `t`.

        Args:
            t (torch.Tensor): Time tensor of shape `(B, 1, 1, 1)` or broadcastable
                to that shape with values in `[0, 1]`.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                `alpha_t`, `sigma_t`, `d_alpha_t`, `d_sigma_t`, all shaped like `t`.
        """
        alpha_t = self.flow_scheduler.alpha(t)
        sigma_t = self.flow_scheduler.sigma(t)
        d_alpha_t = self.flow_scheduler.d_alpha(t)
        d_sigma_t = self.flow_scheduler.d_sigma(t)

        return alpha_t, sigma_t, d_alpha_t, d_sigma_t
    
    def sample_time_steps(self, batch_size, device):
        """Sample a pair of times `(r, t)` with `t >= r`, and set `s = t`.

        Args:
            batch_size (int): Number of samples to draw.
            device (torch.device): Target device for returned tensors.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                `r`, `s`, `t` in `[0, 1]`, each shaped `(B,)`, where `s == t`.
        """
        # Step1: Sample two time points
        if self.time_sampler == "uniform":
            time_samples = torch.rand(batch_size, 2, device=device)
        elif self.time_sampler == "logit_normal":
            normal_samples = torch.randn(batch_size, 2, device=device)
            normal_samples = normal_samples * self.time_sigma + self.time_mu
            time_samples = torch.sigmoid(normal_samples)
        else:
            raise ValueError(f"Unknown time sampler: {self.time_sampler}")
        
        # Step2: Ensure t > r by sorting
        sorted_samples, _ = torch.sort(time_samples, dim=1)
        r, t = sorted_samples[:, 0], sorted_samples[:, 1]
        
        # Step3: Control the proportion of r=t samples
        fraction_equal = 1.0 - self.ratio_r_not_equal_t  # e.g., 0.75 means 75% of samples have r=t
        # Create a mask for samples where r should equal t
        equal_mask = torch.rand(batch_size, device=device) < fraction_equal
        # Apply the mask: where equal_mask is True, set r=t (replace)
        r = torch.where(equal_mask, t, r)
        s = t
        
        return r, s, t 
    
    def __call__(self, model, model_tgt, images, kwargs=None):
        """Compute MeanFlow loss for a batch.

        Args:
            model (Callable): Student model `u(z_t, r, t, **kwargs)` producing velocity.
            model_tgt (Callable): Target/EMA model with the same signature as `model`.
            images (torch.Tensor): Input images `x` of shape `(B, C, H, W)`.
            kwargs (Optional[dict]): Optional model kwargs. If contains key `y`
                (class labels), label dropout may be applied for CFG.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                `loss` of shape `(B,)` (per-sample) and `loss_mean_ref` scalar reference.
        """
        if kwargs == None:
            kwargs = {}
        else:
            kwargs = kwargs.copy()

        batch_size = images.shape[0]
        device = images.device

        unconditional_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
        model_kwargs = {}
        if kwargs.get('y') is not None and self.label_dropout_prob > 0:
            y = kwargs['y'].clone()  
            batch_size = y.shape[0]
            num_classes = model_tgt.num_classes
            dropout_mask = torch.rand(batch_size, device=y.device) < self.label_dropout_prob
            
            y[dropout_mask] = num_classes
            model_kwargs['y'] = y
            unconditional_mask = dropout_mask  # Used for unconditional velocity computation

        # Sample time steps
        r, s, t = self.sample_time_steps(batch_size, device)

        noises = torch.randn_like(images)
        
        # Calculate interpolation and z_t
        alpha_t, sigma_t, d_alpha_t, d_sigma_t = self.interpolant(t.view(-1, 1, 1, 1))
        z_t = alpha_t * images + sigma_t * noises #(1-t) * images + t * noise
        
        # Calculate instantaneous velocity v_t 
        v_t = d_alpha_t * images + d_sigma_t * noises
        
        u = model(z_t, r, t, **model_kwargs)
        
        u_target = self._tgt_u(model_tgt, z_t, v_t, r, s, t, unconditional_mask, **model_kwargs)
                
        loss, loss_mean_ref = self.loss_u(u, u_target)
        return loss, loss_mean_ref

    def loss_u(self, u_pred, u_target):
        """Compute velocity loss between prediction and target.

        Args:
            u_pred (torch.Tensor): Predicted velocity, shape `(B, C, H, W)`.
            u_target (torch.Tensor): Target velocity, shape `(B, C, H, W)`.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                Per-sample loss `(B,)` and mean squared error scalar as reference.
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

    def _tgt_u(self, model_tgt, z_t, v_t, r, s, t, unconditional_mask, **model_kwargs):
        """Compute target velocity `u_target` using JVP and optional CFG.

        In MeanFlow, by construction `s = t`. For samples within the CFG time
        window and with labels present (and not dropped), a mixed velocity is
        used to compute the JVP; otherwise the standard JVP is used.

        Args:
            model_tgt (Callable): Target/EMA model `u(z_t, r, t, **kwargs)`.
            z_t (torch.Tensor): Interpolated state, shape `(B, C, H, W)`.
            v_t (torch.Tensor): Instantaneous velocity d/dt of the interpolant, `(B, C, H, W)`.
            r (torch.Tensor): Start times, shape `(B,)`.
            s (torch.Tensor): End times, equals `t`, shape `(B,)`.
            t (torch.Tensor): Current times, shape `(B,)`.
            unconditional_mask (torch.Tensor): Boolean mask `(B,)` of samples with
                label dropped (unconditional) for CFG.
            **model_kwargs: Optional keyword args forwarded to `model_tgt` (e.g., `y`).

        Returns:
            torch.Tensor: Target velocity `u_target` of shape `(B, C, H, W)`.
        """
        batch_size = z_t.shape[0]
        time_diff = (t - r).view(-1, 1, 1, 1)      
        u_target = torch.zeros_like(v_t)        
        
        # Check if CFG should be applied (exclude unconditional samples)
        cfg_time_mask = (t >= self.cfg_min_t) & (t <= self.cfg_max_t) & (~unconditional_mask)
        
        if model_kwargs.get('y') is not None and cfg_time_mask.any():
            # Split samples into CFG and non-CFG
            cfg_indices = torch.where(cfg_time_mask)[0]
            no_cfg_indices = torch.where(~cfg_time_mask)[0]
            
            u_target = torch.zeros_like(v_t)
            
            # Process CFG samples
            if len(cfg_indices) > 0:
                cfg_z_t = z_t[cfg_indices]
                cfg_v_t = v_t[cfg_indices]
                cfg_r = r[cfg_indices]
                cfg_t = t[cfg_indices]
                cfg_time_diff = time_diff[cfg_indices]
                
                cfg_kwargs = {}
                for k, v in model_kwargs.items():
                    if torch.is_tensor(v) and v.shape[0] == batch_size:
                        cfg_kwargs[k] = v[cfg_indices]
                    else:
                        cfg_kwargs[k] = v
                
                # Compute v_tilde for CFG samples
                cfg_y = cfg_kwargs.get('y')
                num_classes = model_tgt.num_classes
                
                cfg_z_t_batch = torch.cat([cfg_z_t, cfg_z_t], dim=0)
                cfg_t_batch = torch.cat([cfg_t, cfg_t], dim=0)
                cfg_t_end_batch = torch.cat([cfg_t, cfg_t], dim=0)
                cfg_y_batch = torch.cat([cfg_y, torch.full_like(cfg_y, num_classes)], dim=0)
                
                cfg_combined_kwargs = cfg_kwargs.copy()
                cfg_combined_kwargs['y'] = cfg_y_batch
                
                with torch.no_grad():
                    cfg_combined_u_at_t = model_tgt(cfg_z_t_batch, cfg_t_batch, cfg_t_end_batch, **cfg_combined_kwargs)
                    cfg_u_cond_at_t, cfg_u_uncond_at_t = torch.chunk(cfg_combined_u_at_t, 2, dim=0)
                    cfg_v_tilde = (self.cfg_omega * cfg_v_t + 
                            self.cfg_kappa * cfg_u_cond_at_t + 
                            (1 - self.cfg_omega - self.cfg_kappa) * cfg_u_uncond_at_t)
                
                # Compute JVP with CFG velocity
                def fn_current_cfg(z, cur_r, cur_t):
                    return model_tgt(z, cur_r, cur_t, **cfg_kwargs)
                
                primals = (cfg_z_t, cfg_r, cfg_t)
                tangents = (cfg_v_tilde, torch.zeros_like(cfg_r), torch.ones_like(cfg_t))
                _, cfg_dudt = jvp(fn_current_cfg, primals, tangents)
                
                cfg_u_target = cfg_v_tilde - cfg_time_diff * cfg_dudt
                u_target[cfg_indices] = cfg_u_target
            
            # Process non-CFG samples (including unconditional ones)
            if len(no_cfg_indices) > 0:
                no_cfg_z_t = z_t[no_cfg_indices]
                no_cfg_v_t = v_t[no_cfg_indices]
                no_cfg_r = r[no_cfg_indices]
                no_cfg_t = t[no_cfg_indices]
                no_cfg_time_diff = time_diff[no_cfg_indices]
                
                no_cfg_kwargs = {}
                for k, v in model_kwargs.items():
                    if torch.is_tensor(v) and v.shape[0] == batch_size:
                        no_cfg_kwargs[k] = v[no_cfg_indices]
                    else:
                        no_cfg_kwargs[k] = v
                
                def fn_current_no_cfg(z, cur_r, cur_t):
                    return model_tgt(z, cur_r, cur_t, **no_cfg_kwargs)
                
                primals = (no_cfg_z_t, no_cfg_r, no_cfg_t)
                tangents = (no_cfg_v_t, torch.zeros_like(no_cfg_r), torch.ones_like(no_cfg_t))
                _, no_cfg_dudt = jvp(fn_current_no_cfg, primals, tangents)
                
                no_cfg_u_target = no_cfg_v_t - no_cfg_time_diff * no_cfg_dudt
                u_target[no_cfg_indices] = no_cfg_u_target
        else:
            # No labels or no CFG applicable samples, use standard JVP
            primals = (z_t, r, t)
            tangents = (v_t, torch.zeros_like(r), torch.ones_like(t))
            
            def fn_current(z, cur_r, cur_t):
                return model_tgt(z, cur_r, cur_t, **model_kwargs)

            _, dudt = jvp(fn_current, primals, tangents)
            
            u_target = v_t - time_diff * dudt
                
        return u_target