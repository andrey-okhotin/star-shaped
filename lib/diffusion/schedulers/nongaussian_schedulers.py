import os
from pathlib import Path

import torch
import math
import numpy as np
from scipy import interpolate

from saving_utils.get_repo_root import get_repo_root
    
    
    
def nongaussian_scheduler(scheduler, num_steps):
    
    """
    DESCRIPTION:
        Schedulers for different SS-DDPM models.
        
    INPUT:
        <>  scheduler -> str: name of predefined schedule
        <>  num_steps -> int: number of steps in diffusion model
        
    OUTPUT:
        <>  *parameters -> *tuple: parameters for a
    
    """
    
    schedulers = os.path.join(get_repo_root(), 'lib', 'diffusion', 'schedulers')
    def time_rescaler(t):
        return time_rescaler.discretization[t.to(torch.long)]
    time_rescaler.discretization = 999 * torch.linspace(1e-3, 1., num_steps)
    
    
    if   scheduler == 'beta_ss_as_cosine_ddpm':
        nu1 = torch.tensor(
            np.load(os.path.join(schedulers, 'nu_t_as_cosine.npy')),
            dtype=torch.float32
        )        
        nu_t = nu1 + 2
        xi_t = 2 / nu_t
        return nu_t, xi_t, time_rescaler
    
    
    elif scheduler == 'default_dirichlet':
        params = {
            'nu_bounds' : (2000.0, 1e-6),
            'nu_coefs' : [ 1.25 ]
        }
        nu_t = exponential_decay(params['nu_bounds'], params['nu_coefs'], num_steps)
        return nu_t, time_rescaler
    
    
    elif scheduler == 'default_wishart':
        params = {
            'xi_bounds' : (1.0, 1e-4),
            'xi_degree' : 0.5,
            'n_bounds' : (20000.0, 5.),
            'n_coefs' : [ 1.3 ]
        }
        xi_t = degree_decay(
            params['xi_bounds'], params['xi_degree'], num_steps)
        n_t = exponential_decay(
            params['n_bounds'], params['n_coefs'], num_steps)
        return xi_t, n_t, time_rescaler
    
    
    elif scheduler == 'default_d3pm':
        def get_uniform_transition_mat(vocab_size, beta_t):
            mat = torch.full((vocab_size, vocab_size), beta_t/float(vocab_size))
            diag_indices = np.diag_indices_from(mat.numpy())
            diag_val = 1 - beta_t * (vocab_size - 1) / vocab_size
            mat[diag_indices] = diag_val
            return mat
        s = 0.008
        vocab_size = 27
        f_t = lambda t: torch.cos((t / (num_steps+1) + s) / (1 + s) * math.pi / 2)
        a_t = lambda t: f_t(t) / f_t(torch.tensor(0., dtype=torch.float64))
        discretization = torch.arange(1, num_steps+1, dtype=torch.float64)
        cumprod_alphas_t = a_t(discretization)
        cumprod_alphas_t_1 = torch.hstack((torch.tensor([1.0]), cumprod_alphas_t[:-1]))
        betas_t = 1 - cumprod_alphas_t / cumprod_alphas_t_1
        Q_t = []
        for t in range(num_steps):
            Q_t.append(get_uniform_transition_mat(vocab_size, betas_t[t]))
        Q_t = torch.stack(Q_t)
        cumprod_Q_t = [ Q_t[0] ]
        for t in range(1, num_steps):
            cumprod_Q_t.append(torch.matmul(cumprod_Q_t[-1], Q_t[t]))
        cumprod_Q_t = torch.stack(cumprod_Q_t)      
        return betas_t, Q_t, cumprod_Q_t, time_rescaler
    
    
    elif scheduler == 'categorical_ss_as_d3pm':
        vocab_size = 27
        betas_t = np.load(os.path.join(schedulers, 'cum_beta_t.npy'))
        T = betas_t.size
        eyes = np.expand_dims(np.eye(vocab_size), 0).repeat(T, 0)
        ones = np.ones_like(eyes)
        betas_t = betas_t.reshape(T, 1, 1)
        cumprod_Q_t = (1 - betas_t) * eyes + (betas_t / vocab_size) * ones
        cumprod_Q_t = torch.tensor(cumprod_Q_t, dtype=torch.float32)
        return betas_t, None, cumprod_Q_t, time_rescaler


    elif scheduler == 'default_von_mises_fisher':
        T = 100
        fit_params_y = [3, 0.7, 0, -3]
        fit_params_x = np.array([0.01, 0.3, 0.7, 1])
        start_kappa_pow = 3
        kappa_0, kappa_T = np.max(fit_params_y), np.min(fit_params_y)
        fit_y_scaled = (fit_params_y - kappa_T) * ((start_kappa_pow - kappa_T) / (kappa_0 - kappa_T)) + kappa_T
        spline = interpolate.CubicSpline(T * fit_params_x, fit_y_scaled)
        log10_theta = torch.tensor(spline(T * np.arange(num_steps) / num_steps), dtype=torch.float32)
        kappa_t = torch.pow(10, log10_theta)
        return kappa_t, time_rescaler
    
    raise NotImplementedError




def exponential_decay(bounds, coefs, num_steps):
    coefs = coefs[::-1]
    x = [ 1e-6 ]
    segment = num_steps // len(coefs)
    for i in range(num_steps-1):
        x.append(x[-1] * coefs[i//segment])
    x = torch.flip(torch.tensor(x), dims=[0])
    x = x - x[-1]
    x = x * (bounds[0] - bounds[1]) / x[0]
    x = x + bounds[1]
    return x




def degree_decay(bounds, degree, num_steps):
    return torch.linspace(
        bounds[0]**(1./degree), bounds[1]**(1./degree), num_steps, dtype=torch.float32
    ).pow(degree)



