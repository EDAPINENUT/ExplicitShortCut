'''
Adapted from the JAX implementation of SCD (https://github.com/kvfrans/shortcut-models)
'''
import torch
import numpy as np
from utils.scheduler import LinearFlowScheduler

class SCDLoss:
    """ShortCut  Diffusion (SCD) loss.

    Provides a two-branch training objective: a bootstrap (consistency) path
    with multi-hop guidance and a flow-matching path. Supports classifier-free
    guidance (CFG) via label dropout and mixing at specific hops.
    """
    def __init__(
        self,
        path_type="linear",
        loss_type="l2",
        # New parameters
        ratio_r_not_equal_t=0.25,     # Ratio of samples where r≠t
        total_step=128,
        adaptive_p=1.0,               # Power param for adaptive loss
        label_dropout_prob=0.1,       # Drop out label
        # CFG related params
        cfg_omega=1.0,                # CFG omega param, default 1.0 means no CFG
    ):
        """Initialize SCD loss configuration.

        Args:
            path_type (str): Interpolation path type; currently supports "linear".
            loss_type (str): Loss type; "l2" for MSE or "adaptive" for weighted loss.
            ratio_r_not_equal_t (float): Fraction of batch for bootstrap branch
                (consistency path). Remaining goes to flow-matching branch.
            total_step (int): Total discrete steps used to schedule times.
            adaptive_p (float): Power for adaptive weighting when `loss_type` is "adaptive".
            label_dropout_prob (float): Probability to drop labels for CFG (set to
                the extra class index `num_classes`).
            cfg_omega (float): CFG scale for mixing conditional and unconditional velocities.
        """
        self.loss_type = loss_type
        self.path_type = path_type
        
        # Time sampling config
        self.ratio_r_not_equal_t = ratio_r_not_equal_t
        self.total_step = total_step
        self.label_dropout_prob = label_dropout_prob
        # Adaptive weight config
        self.adaptive_p = adaptive_p
        
        # CFG config
        self.cfg_omega = cfg_omega
        if path_type == "linear":
            self.flow_scheduler = LinearFlowScheduler()
        else:
            raise ValueError(f"Unknown path type: {path_type}")
        
    def interpolant(self, t):
        """Compute interpolation scalars and their derivatives at time `t`.

        Args:
            t (torch.Tensor): Time tensor broadcastable to `(B, 1, 1, 1)` in `[0,1]`.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                `alpha_t`, `sigma_t`, `d_alpha_t`, `d_sigma_t`.
        """
        alpha_t = self.flow_scheduler.alpha(t)
        sigma_t = self.flow_scheduler.sigma(t)
        d_alpha_t = self.flow_scheduler.d_alpha(t)
        d_sigma_t = self.flow_scheduler.d_sigma(t)

        return alpha_t, sigma_t, d_alpha_t, d_sigma_t
    
    def sample_time_steps(self, batch_size, device):
        """Sample `(r, s, t)` for bootstrap and flow-matching branches.

        The first `ratio_r_not_equal_t` fraction of the batch forms the bootstrap
        branch with multi-hop times; the rest forms the flow-matching branch where
        `s == r` and `t` is one step ahead.

        Args:
            batch_size (int): Batch size.
            device (torch.device): Target device for returned tensors.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: `r`, `s`, `t` each of
            shape `(B,)` in `[0,1]`.
        """
        # 1) ========= consistency bootstrap time =======
        bootstrap_batchsize = int(batch_size * self.ratio_r_not_equal_t)
        log2_sections = int(np.log2(self.total_step)) # 7
        
        dt_base = torch.arange(log2_sections, 0, -1, device=device) - 1  
        repeats = int(bootstrap_batchsize // log2_sections)  
        dt_base = torch.repeat_interleave(dt_base, repeats)  #[0,1,...6, 0,1,...]
        dt_base = torch.cat([dt_base, torch.zeros(int(bootstrap_batchsize-dt_base.shape[0]), device=device)])
        num_dt_cfg = int(bootstrap_batchsize // log2_sections)  ## 
        
        dt = 1 / (2 ** (dt_base)) # [1/64, 1/32, ... 1, 1] 
        dt_bootstrap = dt / 2 # t-s   s-r
    
        dt_steps = (dt * self.total_step).long()  # [2, 4, 8, ..., 128, 128]
        max_steps = self.total_step
        rand_floats = torch.rand(bootstrap_batchsize, device=device)
        t_steps = (dt_steps + rand_floats * (max_steps - dt_steps + 1)).long()
        t_steps = torch.clamp(t_steps, max=max_steps)

        bst_t = t_steps.float() / self.total_step
        r_steps = t_steps - dt_steps  
        bst_r = r_steps.float() / self.total_step
        s_steps = t_steps - (dt_steps / 2)
        bst_s = s_steps.float() / self.total_step
    
        # 2) ======== flow matching time #######
        fm_t_steps = torch.randint(1, max_steps + 1, size=(batch_size-bootstrap_batchsize,), device=device)
        fm_t = fm_t_steps.float() / self.total_step
        fm_r_steps = fm_t_steps - 1 
        fm_r = fm_r_steps.float() / self.total_step
        fm_s = fm_r
        
        t = torch.concatenate((bst_t, fm_t), dim=0)
        r = torch.concatenate((bst_r, fm_r), dim=0)
        s = torch.concatenate((bst_s, fm_s), dim=0)
        
        return r, s, t
    
    def _tgt_u(self, model_tgt, z_t, v_t, r, s, t, **model_kwargs):
        """Compute target velocity `u_target` for SCD.

        Bootstrap branch computes two guided hops (`t->s` and `s->r`) with CFG
        mixing. Flow-matching branch uses the instantaneous velocity `v_t`.

        Args:
            model_tgt (Callable): Target/EMA model to query velocities.
            z_t (torch.Tensor): Interpolated states, `(B, C, H, W)`.
            v_t (torch.Tensor): Instantaneous interpolant velocity, `(B, C, H, W)`.
            r (torch.Tensor): Start times `(B,)`.
            s (torch.Tensor): Mid times `(B,)`.
            t (torch.Tensor): End times `(B,)`.
            **model_kwargs: Optional kwargs passed to `model_tgt` (e.g., `y`).

        Returns:
            torch.Tensor: Target velocity `(B, C, H, W)` combining branches.
        """
        if model_kwargs == None:
            model_kwargs = {}
        else:
            model_kwargs = model_kwargs.copy()
            
        if model_kwargs.get('y') is not None and self.label_dropout_prob > 0:
            y = model_kwargs['y'].clone()  
            num_classes = model_tgt.num_classes
                   
        batch_size = z_t.shape[0]
        bootstrap_batchsize = int(batch_size * self.ratio_r_not_equal_t)
        log2_sections = int(np.log2(self.total_step))
        num_dt_cfg = int(bootstrap_batchsize // log2_sections)
        device=y.device
            
        ### bst
        bst_y = y[:bootstrap_batchsize]
        bst_t = t[:bootstrap_batchsize]
        bst_s = s[:bootstrap_batchsize]
        bst_r = r[:bootstrap_batchsize]
        bst_z_t = z_t[:bootstrap_batchsize]
        
        bst_z_t_extra = torch.concatenate((bst_z_t, bst_z_t[:num_dt_cfg]), dim=0)
        bst_t_extra = torch.concatenate((bst_t, bst_t[:num_dt_cfg]), dim=0)
        bst_s_extra = torch.concatenate((bst_s, bst_s[:num_dt_cfg]), dim=0)
        bst_r_extra = torch.concatenate((bst_r, bst_r[:num_dt_cfg]), dim=0)
        bst_y_extra = torch.concatenate((bst_y, torch.ones(num_dt_cfg, device=device) * num_classes), dim=0)
        
        ## t -> s 
        v_b1_raw = model_tgt(bst_z_t_extra, bst_s_extra, bst_t_extra, bst_y_extra.long())
        v_b_cond = v_b1_raw[:bootstrap_batchsize]
        v_b_uncond = v_b1_raw[bootstrap_batchsize:]
        v_cfg = v_b_uncond + self.cfg_omega * (v_b_cond[:num_dt_cfg] - v_b_uncond)
        v_b1 = torch.concatenate((v_cfg, v_b_cond[num_dt_cfg:]), dim=0)
        
        ## s -> r
        bst_z_s = bst_z_t + (bst_s-bst_t).view(-1, 1, 1, 1)*v_b1
        bst_z_s = torch.clamp(bst_z_s, -4, 4)
        bst_z_s_extra = torch.concatenate((bst_z_s, bst_z_s[:num_dt_cfg]), dim=0)
        v_b2_raw = model_tgt(bst_z_s_extra, bst_r_extra, bst_s_extra, bst_y_extra.long())
        v_b2_cond = v_b2_raw[:bootstrap_batchsize]
        v_b2_uncond = v_b2_raw[bootstrap_batchsize:]
                    
        v_b2_cfg = v_b2_uncond + self.cfg_omega * (v_b2_cond[:num_dt_cfg] - v_b2_uncond)
        v_b2 = torch.concatenate((v_b2_cfg, v_b2_cond[num_dt_cfg:]), dim=0)
        v_target = (v_b1 + v_b2) / 2
        v_target = torch.clamp(v_target, -4, 4)    
        bst_v = v_target     
                
        ### fm      
        fm_y = y[bootstrap_batchsize:]
        dropout_mask = torch.rand(batch_size-bootstrap_batchsize, device=device) < self.label_dropout_prob
        fm_y[dropout_mask] = num_classes
        fm_t = t[bootstrap_batchsize:]
        fm_r = r[bootstrap_batchsize:]
        fm_z_t = z_t[bootstrap_batchsize:]
        fm_v_t = v_t[bootstrap_batchsize:]       
        
        ### merge
        u_target = torch.concatenate((bst_v, fm_v_t), dim=0)
        
        return u_target
                
    def __call__(self, model, model_tgt, images, model_kwargs=None):
        """Compute SCD loss for a batch.

        Args:
            model (Callable): Student model `u(z_t, r, t, **kwargs)`.
            model_tgt (Callable): Target/EMA model with same signature as `model`.
            images (torch.Tensor): Input images `(B, C, H, W)`.
            model_kwargs (Optional[dict]): Optional kwargs passed to models (e.g., `y`).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Per-sample loss `(B,)` and mean reference.
        """
        batch_size = images.shape[0]
        device = images.device
            
        r, s, t = self.sample_time_steps(batch_size, device)
        noises = torch.randn_like(images)
        # Calculate interpolation and z_t
        alpha_t, sigma_t, d_alpha_t, d_sigma_t = self.interpolant(t.view(-1, 1, 1, 1))
        z_t = alpha_t * images + sigma_t * noises
        v_t = d_alpha_t * images + d_sigma_t * noises
        # loss
        u = model(z_t, r, t, **model_kwargs)
        u_target = self._tgt_u(model_tgt, z_t, v_t, r, s, t, **model_kwargs)
                
        loss, loss_mean_ref = self.loss_u(u, u_target)
        
        return loss, loss_mean_ref
    
    def loss_u(self, u_pred, u_target):
        """Compute velocity loss between prediction and target.

        Args:
            u_pred (torch.Tensor): Predicted velocity `(B, C, H, W)`.
            u_target (torch.Tensor): Target velocity `(B, C, H, W)`.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Per-sample loss `(B,)` and mean reference.
        """
        # Detach the target to prevent gradient flow        
        error = u_pred - u_target.detach()
        loss_mid = torch.sum((error**2).reshape(error.shape[0],-1), dim=-1)
        # Apply adaptive loss based on configuration
        if self.loss_type == "adaptive":
            weights = 1.0 / (loss_mid.detach() ** 2 + 1e-3).pow(self.adaptive_p)
            loss = weights * loss_mid ** 2          
        elif self.loss_type == "l2":
            loss = loss_mid
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        loss_mean_ref = torch.mean((error**2))

        return loss, loss_mean_ref
