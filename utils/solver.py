import torch
from .utils import append_dims

def ddim_solver_condv(z, v, t_start, t_end, alpha_bar, beta_bar):
    alpha_bar_ = alpha_bar(t_start, t_end)
    beta_bar_ = beta_bar(t_start, t_end)
    alpha_bar_ = append_dims(alpha_bar_, z.ndim)
    beta_bar_ = append_dims(beta_bar_, z.ndim)
    z_end = alpha_bar_ * z + beta_bar_ * v
    return z_end


def ddim_solver_condx0(z, x0, t_start, t_end, alpha_hat, beta_hat):
    alpha_hat_ = alpha_hat(t_start, t_end)
    beta_hat_ = beta_hat(t_start, t_end)
    alpha_hat_ = append_dims(alpha_hat_, z.ndim)
    beta_hat_ = append_dims(beta_hat_, z.ndim)
    z_end = alpha_hat_ * z + beta_hat_ * x0
    return z_end


def euler_solver_condx0(xt, x0, t, s, sigma_max=1.0):
    dims = xt.ndim
    d = (xt - x0) / (append_dims(t, dims) * sigma_max)
    samples = xt + d * append_dims(s - t, dims) * sigma_max
    return samples


def euler_solver_condv(xt, v, t, s):
    dims = xt.ndim
    samples = xt + v * append_dims(s - t, dims)
    return samples
