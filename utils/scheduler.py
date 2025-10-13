import torch
import math
import os
import torch.nn.functional as F


class FlowScheduler:
    def __init__(self, ):
        pass

    def alpha(self, t):
        # for z sampling
        pass

    def sigma(self, t):
        # for z sampling
        pass

    def d_alpha(self, t):
        # for v sampling
        pass

    def d_sigma(self, t):
        # for v sampling
        pass

    def alpha_bar(self, t_start, t_end):
        # for ddim sampling conditioned on v
        alpha_s = self.alpha(t_start)
        alpha_t = self.alpha(t_end)
        sigma_s = self.sigma(t_start)
        sigma_t = self.sigma(t_end)
        d_alpha_s = self.d_alpha(t_start)
        d_sigma_s = self.d_sigma(t_start)
        numerator = alpha_t * d_sigma_s - sigma_t * d_alpha_s
        denominator = alpha_s * d_sigma_s - sigma_s * d_alpha_s
        alpha_bar = numerator / denominator
        return alpha_bar
    
    def beta_bar(self, t_start, t_end):
        # for ddim sampling conditioned on v
        alpha_s = self.alpha(t_start)
        alpha_t = self.alpha(t_end)
        sigma_s = self.sigma(t_start)
        sigma_t = self.sigma(t_end)
        d_alpha_s = self.d_alpha(t_start)
        d_sigma_s = self.d_sigma(t_start)
        numerator = -alpha_t * sigma_s + sigma_t * alpha_s
        denominator = alpha_s * d_sigma_s - sigma_s * d_alpha_s
        beta_bar = numerator / denominator
        return beta_bar
    
    def c_in(self, t):
        return torch.ones_like(t)
    
    def c_out(self, t):
        return torch.ones_like(t)
    
    @property
    def sigma_0(self,):
        return 1.0
    
class CosineFlowScheduler(FlowScheduler):
    def __init__(self, sigma_data=0.5):
        self.sigma_data = sigma_data

    def alpha(self, t):
        return torch.cos(t * math.pi / 2)

    def sigma(self, t):
        return torch.sin(t * math.pi / 2)

    def d_alpha(self, t):
        return -math.pi / 2 * torch.sin(t * math.pi / 2)

    def d_sigma(self, t):
        return math.pi / 2 * torch.cos(t * math.pi / 2)

    def alpha_bar(self, t_start, t_end):
        return torch.cos((t_end - t_start) * math.pi / 2)

    def beta_bar(self, t_start, t_end):
        return torch.sin((t_end - t_start) * math.pi / 2) * 2 / math.pi
    
    def d_alpha_bar_dt(self, t_start, t_end):
        # d/dt_start[cos((t_end-t_start) * π/2)] = sin((t_end-t_start) * π/2) * π/2
        return torch.sin((t_end - t_start) * math.pi / 2) * math.pi / 2
    
    def d_beta_bar_dt(self, t_start, t_end):
        # d/dt_start[sin((t_end-t_start) * π/2) * 2/π] = -cos((t_end-t_start) * π/2)
        return -torch.cos((t_end - t_start) * math.pi / 2)
    
    def alpha_hat(self, t_start, t_end):
        # α̂ = σ(s) / σ(t) = sin(πs/2) / sin(πt/2)
        # t_start = t, t_end = s
        return torch.sin(t_end * math.pi / 2) / torch.sin(t_start * math.pi / 2)

    def beta_hat(self, t_start, t_end):
        # β̂ = α(s) - σ(s)/σ(t) * α(t) = cos(πs/2) - sin(πs/2)/sin(πt/2) * cos(πt/2)
        # t_start = t, t_end = s
        alpha_hat = self.alpha_hat(t_start, t_end)
        return self.alpha(t_end) - alpha_hat * self.alpha(t_start)
    
    def c_in(self, t):
        return 1 / self.sigma_data * torch.ones_like(t)
    
    def c_out(self, t):
        return self.sigma_data * torch.ones_like(t)

    @property
    def sigma_0(self,):
        return self.sigma_data



class CosineFlowSchedulerLegacy(FlowScheduler):
    def __init__(self, sigma_data=0.5):
        self.sigma_data = sigma_data

    def alpha(self, t):
        return torch.cos(t)

    def sigma(self, t):
        return torch.sin(t)

    def d_alpha(self, t):
        return - torch.sin(t)

    def d_sigma(self, t):
        return torch.cos(t)

    def alpha_bar(self, t_start, t_end):
        return torch.cos((t_end - t_start))

    def beta_bar(self, t_start, t_end):
        return torch.sin((t_end - t_start)) 
    
    def d_alpha_bar_dt(self, t_start, t_end):
        # d/dt_start[cos((t_end-t_start) * π/2)] = sin((t_end-t_start) * π/2) * π/2
        return torch.sin((t_end - t_start))
    
    def d_beta_bar_dt(self, t_start, t_end):
        # d/dt_start[sin((t_end-t_start) * π/2) * 2/π] = -cos((t_end-t_start) * π/2)
        return -torch.cos((t_end - t_start))
    
    def alpha_hat(self, t_start, t_end):
        # α̂ = σ(s) / σ(t) = sin(πs/2) / sin(πt/2)
        # t_start = t, t_end = s
        return torch.sin(t_end) / torch.sin(t_start)

    def beta_hat(self, t_start, t_end):
        # β̂ = α(s) - σ(s)/σ(t) * α(t) = cos(πs/2) - sin(πs/2)/sin(πt/2) * cos(πt/2)
        # t_start = t, t_end = s
        alpha_hat = self.alpha_hat(t_start, t_end)
        return self.alpha(t_end) - alpha_hat * self.alpha(t_start)
    
    def c_in(self, t):
        return 1 / self.sigma_data * torch.ones_like(t)
    
    def c_out(self, t):
        return self.sigma_data * torch.ones_like(t)

    @property
    def sigma_0(self,):
        return self.sigma_data

    
class LinearFlowScheduler(FlowScheduler):
    def __init__(self, sigma_data=1.0):
        super().__init__()
        self.sigma_data = sigma_data

    def alpha(self, t):
        return 1 - t
    
    def sigma(self, t):
        return t
    
    def d_alpha(self, t):
        return -torch.ones_like(t)
    
    def d_sigma(self, t):
        return torch.ones_like(t)
    
    def alpha_bar(self, t_start, t_end):
        return torch.ones_like(t_start)
    
    def beta_bar(self, t_start, t_end):
        return t_end - t_start
    
    def alpha_hat(self, t_start, t_end):
        return t_end / t_start
    
    def beta_hat(self, t_start, t_end):
        return (t_start - t_end) / t_start
    
    def c_in(self, t):
        return torch.ones_like(t) * 1 / self.sigma_data
    
    def c_out(self, t):
        return torch.ones_like(t) * self.sigma_data
    
    def d_alpha_bar_dt(self, t_start, t_end):
        return torch.zeros_like(t_start)
    
    def d_beta_bar_dt(self, t_start, t_end):
        return - torch.ones_like(t_start)
    
    @property
    def sigma_0(self,):
        return self.sigma_data
    
    

class EDMFlowScheduler(FlowScheduler):
    def __init__(self, sigma_min, sigma_max, sigma_data):
        super().__init__()
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data

    def alpha(self, t):
        return torch.ones_like(t)
    
    def sigma(self, t):
        return t * self.sigma_max
    
    def d_alpha(self, t):
        return torch.zeros_like(t)
    
    def d_sigma(self, t):
        return self.sigma_max

    def c_in(self, t):
        sigma = t * self.sigma_max
        c_in = 1 / (sigma**2 + self.sigma_data**2) ** 0.5
        return c_in
    
    def c_out(self, t):
        return torch.ones_like(t)
    
    def alpha_bar(self, t_start, t_end=0):
        sigma = t_start * self.sigma_max
        c_skip = self.sigma_data**2 / (
            (sigma - self.sigma_min) ** 2 + self.sigma_data**2
        ) 
        return c_skip
    
    def beta_bar(self, t_start, t_end=0):
        sigma = t_start * self.sigma_max
        c_out = (
            (sigma - self.sigma_min)
            * self.sigma_data
            / (sigma**2 + self.sigma_data**2) ** 0.5
        )
        return c_out

    @property
    def sigma_0(self,):
        return self.sigma_max
    
        