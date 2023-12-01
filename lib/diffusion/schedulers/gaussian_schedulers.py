import torch
import math


    
def gaussian_scheduler(scheduler, num_steps):
    """
    """ 
    if   scheduler == 'cosine':
        s = 0.008
        f_t = lambda t: torch.cos((t / (num_steps+1) + s) / (1 + s) * math.pi / 2)**2
        a_t = lambda t: f_t(t) / f_t(torch.tensor(0., dtype=torch.float64))
        discretization = torch.arange(1, num_steps+1, dtype=torch.float64)
        cumprod_alphas_t = a_t(discretization)
        
        
    elif scheduler == 'gauss_ss_as_cosine_ddpm':
        s = 0.008
        f_t = lambda t: torch.cos((t / (num_steps+1) + s) / (1 + s) * math.pi / 2)**2
        a_t = lambda t: f_t(t) / f_t(torch.tensor(0., dtype=torch.float64))
        discretization = torch.arange(1, num_steps+1, dtype=torch.float64)
        cumprod_alphas_t = a_t(discretization)
        first_cumprod_alpha_t = cumprod_alphas_t[0]
        last_cumprod_alphas_t = cumprod_alphas_t[-1]
        cumprod_alphas_t_1 = torch.hstack((
            torch.tensor([(1.0+first_cumprod_alpha_t)/2]), cumprod_alphas_t[:-1]
        ))
        K = ( cumprod_alphas_t_1 / (1 - cumprod_alphas_t_1) -
              cumprod_alphas_t / (1 - cumprod_alphas_t) )
        cumprod_alphas_t_1 = K / (1 + K)
        cumprod_alphas_t = torch.hstack((
            cumprod_alphas_t_1[1:], torch.tensor([last_cumprod_alphas_t])
        ))
    
    
    def time_rescaler(t):
        return time_rescaler.discretization[t.to(torch.long)]
    time_rescaler.discretization = 1000 / num_steps * discretization.to(torch.float32)
    cumprod_alphas_t = cumprod_alphas_t.to(torch.float32)
    return cumprod_alphas_t, time_rescaler
