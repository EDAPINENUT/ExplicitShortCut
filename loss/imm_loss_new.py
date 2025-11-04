import torch
import numpy as np
from utils.scheduler import LinearFlowScheduler
from einops import rearrange
from utils.solver import ddim_solver_condv
from utils import append_dims

class IMMLoss:
    def __init__(
        self,
        path_type="linear",
        # New parameters
        time_sampler="uniform",  # Time sampling strategy: "uniform" or "logit_normal"
        label_dropout_prob=0.1,
        group_size=4,
        gamma=12
    ):
        self.path_type = path_type
        
        # Time sampling config
        self.time_sampler = time_sampler
        self.label_dropout_prob = label_dropout_prob
        self.gamma = gamma
        self.group_size = group_size
        self.sigma_data = 0.5
        if path_type == "linear":
            self.flow_scheduler = LinearFlowScheduler(sigma_data=self.sigma_data)
        else:
            raise ValueError("Unsupported path type: {}".format(path_type))
        

    def interpolant(self, t):
        """Define interpolation function"""
        alpha_t = self.flow_scheduler.alpha(t)
        sigma_t = self.flow_scheduler.sigma(t)
        d_alpha_t = self.flow_scheduler.d_alpha(t)
        d_sigma_t = self.flow_scheduler.d_sigma(t)

        return alpha_t, sigma_t, d_alpha_t, d_sigma_t
    
    def sample_time_steps(self, batch_size, device):
        """Sample time steps (r, t) according to the configured sampler"""

        def get_log_nt(t):
            dtype = t.dtype
            t = t.to(torch.float64)
            logsnr = 2 * ((1 - t).log() - t.log())        
            logsnr = logsnr.to(dtype)
            return -0.5 * logsnr
        
        def nt_to_t(nt):
            t = nt / (1 + nt)
            return t

        t_max, t_min = 0.9940, 0.0
        nt_max_min = get_log_nt(torch.tensor([t_max, t_min])).exp()
        nt_max, nt_min = nt_max_min[0].item(), nt_max_min[1].item()

        if self.time_sampler == "uniform":
            t = torch.rand(batch_size, device=device) * (t_max - t_min) + t_min
            nt = get_log_nt(t).exp()

            r_max = nt_to_t(nt)
            r = torch.rand(batch_size, device=device) * (r_max - t_min) + t_min
            r = torch.minimum(r, t).clamp(min=t_min)

            u = (nt_max - nt_min) * (1 / 2) ** self.gamma
            ns = (nt - u).clamp(min=nt_min, max=nt_max)
            s = nt_to_t(ns)
        else:
            raise ValueError("Unsupported time sampler: {}".format(self.time_sampler))
        
        return r, s, t 

    
    def __call__(self, model, model_tgt, images, kwargs=None):
        """
        Compute MeanFlow loss function with bootstrap mechanism
        """
        model_kwargs = {}

        batch_size = images.shape[0]
        device = images.device
        group_size = self.group_size
        group_num = batch_size // group_size

        
        if kwargs.get('y') is not None and self.label_dropout_prob > 0:
            y = kwargs['y'].clone()  
            batch_size = y.shape[0]
            num_classes = model_tgt.num_classes
            dropout_mask = torch.rand(batch_size, device=y.device) < self.label_dropout_prob
            
            y[dropout_mask] = num_classes
            model_kwargs['y'] = y
            unconditional_mask = dropout_mask  # Used for unconditional velocity computation

        # Sample time steps
        r, s, t = self.sample_time_steps(group_num, device)
        r = r.repeat_interleave(group_size, dim=0)
        s = s.repeat_interleave(group_size, dim=0)
        t = t.repeat_interleave(group_size, dim=0)

        noises = torch.randn_like(images) * self.flow_scheduler.sigma_data
        c_in, c_out = append_dims(self.flow_scheduler.c_in(t), noises.ndim), append_dims(self.flow_scheduler.c_out(t), noises.ndim)
        
        # Calculate interpolation and z_t
        alpha_t, sigma_t, d_alpha_t, d_sigma_t = self.interpolant(t.view(-1, 1, 1, 1))
        z_t = alpha_t * images + sigma_t * noises #(1-t) * images + t * noise
        alpha_s, sigma_s, d_alpha_s, d_sigma_s = self.interpolant(s.view(-1, 1, 1, 1))
        z_s = alpha_s * images + sigma_s * noises
        
        u_tr = model(z_t*c_in, r, t, **model_kwargs) * c_out
        z_tr = ddim_solver_condv(z_t, u_tr, t, r, self.flow_scheduler.alpha_bar, self.flow_scheduler.beta_bar)

        with torch.no_grad(): # disable fp16
            with torch.amp.autocast('cuda', enabled=False):
                u_sr = model(z_s*c_in, r, s, **model_kwargs)*c_out
                z_sr = ddim_solver_condv(z_s, u_sr, s, r, self.flow_scheduler.alpha_bar, self.flow_scheduler.beta_bar)
            
        z_tr = rearrange(z_tr, "(b m) ... -> b m ...", m=group_size)
        z_sr = rearrange(z_sr, "(b m) ... -> b m ...", m=group_size) 
        z_t = rearrange(z_t, "(b m) ... -> b m ...", m=group_size)
        z_s = rearrange(z_s, "(b m) ... -> b m ...", m=group_size)
        t = rearrange(t, "(b m) ... -> b m ...", m=group_size)
        r = rearrange(r, "(b m) ... -> b m ...", m=group_size) 
        s = rearrange(s, "(b m) ... -> b m ...", m=group_size)
                        
        loss, loss_mean_ref = self.kernel_loss(z_tr, z_sr, t, r)
        return loss, loss_mean_ref

    def kernel_fn(self, x, y, flatten_dim, w ):
        x = x.unsqueeze(2)
        y = y.unsqueeze(1)
        w = append_dims(w, x.ndim)
        loss = (
                torch.clamp_min(
                    ((x - y) ** 2).flatten(flatten_dim).sum(-1)  , 1e-8
                )
            ).sqrt()   / (np.prod(y.shape[flatten_dim:]))
            
            
        ret = torch.exp( -loss * w ) 
        return ret 

    def kernel_loss(self, z_tr, z_sr, t, s):
        """
        Compute loss for velocity u
        """
        w, wout = self.get_kernel_weight(t[:, 0], s[:, 0])

        inter_sample_sim = self.kernel_fn(
            z_tr,
            z_tr, 
            w=w,
            flatten_dim=3
        ) 
        inter_tgt_sim = self.kernel_fn(
            z_sr,
            z_sr, 
            w=w,
            flatten_dim=3
        )

        cross_sim = self.kernel_fn(
            z_tr,
            z_sr, 
            w=w,
            flatten_dim=3
        ) 

        inter_sample_sim = inter_sample_sim.mean((1, 2)) 
        cross_sim = cross_sim.mean((1, 2))
        inter_tgt_sim = inter_tgt_sim.mean((1, 2))
            
        loss = inter_sample_sim + inter_tgt_sim - 2 * cross_sim 

        if wout is not None:
            wout = append_dims(wout, loss.ndim)
            loss_out = wout * loss
        else:
            loss_out = loss

        return loss_out, loss

    def get_kernel_weight(self, t, r, a=2, b=4):

        def get_logsnr(t):
            dtype = t.dtype
            t = t.to(torch.float64)

            logsnr = 2 * ((1 - t).log() - t.log())
                
            logsnr = logsnr.to(dtype)
            return logsnr

        def get_logsnr_prime(t):
            return -2 * (1 / (1 - t) / t)

        logsnr_t = get_logsnr(t)
        alpha_t, sigma_t, d_alpha_t, d_sigma_t = self.interpolant(t)
        alpha_r, sigma_r, d_alpha_r, d_sigma_r = self.interpolant(r)

        w = 1 / ((t - r).abs() * self.flow_scheduler.sigma_data + 1e-6)

        neg_dlogsnr_dt = - get_logsnr_prime(t)
                
        wout =  alpha_t ** a / (alpha_t**2 + sigma_t**2)  * 0.5 * neg_dlogsnr_dt * (b - logsnr_t).sigmoid()
        return w, wout
        

    def _tgt_u(self, model_tgt, z_t, v_t, r, s, t, unconditional_mask, **model_kwargs):
        """
        Compute target velocity u_target for CFG samples
        """
        pass
