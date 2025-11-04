import torch
import numpy as np
from torch.func import jvp
from utils.scheduler import LinearFlowScheduler

class ESCLoss:
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
        grad_warmup_steps=10000,
        # V_Plugin related params
        use_vplug=True,
        vplug_select_prob=0.2,
        variational_adaptive_weight=False
    ):
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
        self.use_vplug = use_vplug
        
        # CFG config
        self.cfg_omega = cfg_omega
        self.cfg_kappa = cfg_kappa
        self.cfg_min_t = cfg_min_t
        self.cfg_max_t = cfg_max_t
        self.grad_warmup_steps = grad_warmup_steps
        if path_type == "linear":
            self.flow_scheduler = LinearFlowScheduler()
        self.vplug_select_prob = vplug_select_prob
        self.variational_adaptive_weight = variational_adaptive_weight

    def interpolant(self, t):
        """Define interpolation function"""
        alpha_t = self.flow_scheduler.alpha(t)
        sigma_t = self.flow_scheduler.sigma(t)
        d_alpha_t = self.flow_scheduler.d_alpha(t)
        d_sigma_t = self.flow_scheduler.d_sigma(t)

        return alpha_t, sigma_t, d_alpha_t, d_sigma_t
    
    def sample_time_steps(self, batch_size, step_r, device):
        """Sample time steps (r, t) according to the configured sampler"""
        # Step1: Sample two time points
        if self.time_sampler == "uniform":
            time_samples = torch.rand(batch_size, 2, device=device)
        elif self.time_sampler == "logit_normal":
            normal_samples = torch.randn(batch_size, 2, device=device)
            normal_samples = normal_samples * self.time_sigma + self.time_mu
            time_samples = torch.sigmoid(normal_samples)
        else:
            raise ValueError(f"Unknown time sampler: {self.time_sampler}")
        
        def arccos_scheduler(x, start=1.0, end=0.0):
            if x <= 1.0:
                return end + (start - end) * (np.arccos(x) / np.arccos(0))
            else:
                return end

        # Step2: Ensure t > r by sorting
        sorted_samples, _ = torch.sort(time_samples, dim=1)
        r, t = sorted_samples[:, 0], sorted_samples[:, 1]
        
        fractional_terminate = arccos_scheduler(step_r)
        zero_mask = torch.rand(batch_size, device=device) < fractional_terminate
        r = torch.where(zero_mask, torch.zeros_like(r), r)
        
        # Step3: Control the proportion of r=t samples
        fraction_equal = 1.0 - self.ratio_r_not_equal_t  # e.g., 0.75 means 75% of samples have r=t
        # Create a mask for samples where r should equal t
        equal_mask = torch.rand(batch_size, device=device) < fraction_equal
        # Apply the mask: where equal_mask is True, set r=t (replace)
        r = torch.where(equal_mask, t, r)
        s = t
        
        return r, s, t 
    
    def __call__(self, model, model_tgt, images, kwargs=None):
        """
        Compute MeanFlow loss function with bootstrap mechanism
        """
        if kwargs == None:
            kwargs = {}
        else:
            kwargs = kwargs.copy()
        
        step_r = kwargs.get('step_r', 1.0)
        batch_size = images.shape[0]
        device = images.device

        unconditional_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        model_kwargs = kwargs.copy()
        if model_kwargs.get('y') is not None and self.label_dropout_prob > 0:
            y = model_kwargs['y'].clone()  
            batch_size = y.shape[0]
            num_classes = model_tgt.num_classes
            dropout_mask = torch.rand(batch_size, device=y.device) < self.label_dropout_prob
            
            y[dropout_mask] = num_classes
            model_kwargs['y'] = y
            unconditional_mask = dropout_mask  # Used for unconditional velocity computation
        
        # Sample time steps
        r, s, t = self.sample_time_steps(batch_size, step_r, device)

        noises = torch.randn_like(images)
        
        # Calculate interpolation and z_t
        alpha_t, sigma_t, d_alpha_t, d_sigma_t = self.interpolant(t.view(-1, 1, 1, 1))
        z_t = alpha_t * images + sigma_t * noises #(1-t) * images + t * noise
        
        # Calculate instantaneous velocity v_t 
        v_t = d_alpha_t * images + d_sigma_t * noises
        if self.use_vplug:
            vplug_t = self._estimate_vplug(images, alpha_t, sigma_t, d_alpha_t, d_sigma_t, z_t)
        else:
            vplug_t = v_t
        
        vplug_select = torch.rand(batch_size, device=device) < self.vplug_select_prob
        vplug_t = torch.where(vplug_select.view(-1, 1, 1, 1), vplug_t, v_t)
        if self.variational_adaptive_weight:
            u, log_var = model(z_t, r, t, model_kwargs['y'], return_logvar=True)
        else:
            u = model(z_t, r, t, model_kwargs['y'])
            log_var = torch.zeros(batch_size, device=device)
        
        u_target = self._tgt_u(model_tgt, z_t, vplug_t, r, s, t, unconditional_mask, **model_kwargs)
                
        loss, loss_mean_ref = self.loss_u(u, u_target)
        loss = 1 / torch.exp(log_var) * loss + log_var
        return loss, loss_mean_ref

    def _estimate_vplug(self, images, alpha_t, sigma_t, d_alpha_t, d_sigma_t, z_t):
        images_expand = images.unsqueeze(1)
        z_t_expand = z_t.unsqueeze(0)
        alpha_t = alpha_t.unsqueeze(0)
        sigma_t = sigma_t.unsqueeze(0)
        d_alpha_t = d_alpha_t.unsqueeze(0)
        d_sigma_t = d_sigma_t.unsqueeze(0)
        eps_expand = (z_t_expand - images_expand * alpha_t) / sigma_t
        normal = torch.distributions.Normal(loc=0.0, scale=1.0)
        kernel_weight = torch.softmax(
            torch.sum(normal.log_prob(eps_expand), dim=(-1,-2,-3)), 
            dim=0
        )
        v_t_expand = d_alpha_t * images_expand + d_sigma_t * eps_expand
        vplug = torch.einsum("bvclw,bv->vclw", v_t_expand, kernel_weight)
        return vplug


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

    def _tgt_u(self, model_tgt, z_t, v_t, r, s, t, unconditional_mask, **model_kwargs):
        """
        Compute target velocity u_target for CFG samples
        """
        batch_size = z_t.shape[0]
        time_diff = (t - r).view(-1, 1, 1, 1)      
        u_target = torch.zeros_like(v_t)        
        
        # Check if CFG should be applied (exclude unconditional samples)
        cfg_time_mask = (t >= self.cfg_min_t) & (t <= self.cfg_max_t) & (~unconditional_mask)
        if model_kwargs.get('global_step') is not None and self.grad_warmup_steps > 0:
            warmup_ratio = model_kwargs.get('global_step') / self.grad_warmup_steps
            warmup_ratio = min(warmup_ratio, 1.0)
        else:
            warmup_ratio = 1.0

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
                    cfg_combined_u_at_t = model_tgt(cfg_z_t_batch, cfg_t_batch, cfg_t_end_batch, cfg_combined_kwargs['y'])
                    cfg_u_cond_at_t, cfg_u_uncond_at_t = torch.chunk(cfg_combined_u_at_t, 2, dim=0)
                    cfg_v_tilde = (self.cfg_omega * cfg_v_t + 
                            self.cfg_kappa * cfg_u_cond_at_t + 
                            (1 - self.cfg_omega - self.cfg_kappa) * cfg_u_uncond_at_t)
                
                # Compute JVP with CFG velocity
                def fn_current_cfg(z, cur_r, cur_t):
                    return model_tgt(z, cur_r, cur_t, cfg_kwargs['y'])
                
                primals = (cfg_z_t, cfg_r, cfg_t)
                tangents = (cfg_v_tilde, torch.zeros_like(cfg_r), torch.ones_like(cfg_t))
                _, cfg_dudt = jvp(fn_current_cfg, primals, tangents)
                
                cfg_u_target = cfg_v_tilde - cfg_time_diff * cfg_dudt * warmup_ratio
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
                    return model_tgt(z, cur_r, cur_t, no_cfg_kwargs['y'])
                
                primals = (no_cfg_z_t, no_cfg_r, no_cfg_t)
                tangents = (no_cfg_v_t, torch.zeros_like(no_cfg_r), torch.ones_like(no_cfg_t))
                _, no_cfg_dudt = jvp(fn_current_no_cfg, primals, tangents)
                
                no_cfg_u_target = no_cfg_v_t - no_cfg_time_diff * no_cfg_dudt * warmup_ratio
                u_target[no_cfg_indices] = no_cfg_u_target
        else:
            # No labels or no CFG applicable samples, use standard JVP
            primals = (z_t, r, t)
            tangents = (v_t, torch.zeros_like(r), torch.ones_like(t))
            
            def fn_current(z, cur_r, cur_t):
                return model_tgt(z, cur_r, cur_t, **model_kwargs)

            _, dudt = jvp(fn_current, primals, tangents)
            
            u_target = v_t - time_diff * dudt * warmup_ratio
                
        return u_target