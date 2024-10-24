import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

from diffusion.general_diffusion.markov_diffusion import MarkovDiffusion
from diffusion.schedulers.nongaussian_schedulers import nongaussian_scheduler



class D3PM(MarkovDiffusion):
    
    """
    DESCRIPTION:
        Markovian diffusion model with Categorical distribution. Used for experiments on 
        Text8. 
        
    """

    def __init__(self, diffusion_config):
        super().__init__(diffusion_config)
        _, Q_t, cumprod_Q_t, self.time_rescaler = nongaussian_scheduler(
            diffusion_config.scheduler, self.num_steps)
        
        Q_t_1 = torch.vstack((torch.eye(*Q_t[0].shape)[None,...], Q_t[:-1]))
        self.Q_t = Q_t[:,None,:,:]
        self.Q_t_1 = Q_t_1[:,None,:,:]
        cumprod_Q_t_1 = torch.vstack((
            torch.eye(*cumprod_Q_t[0].shape)[None,...], cumprod_Q_t[:-1]
        ))
        self.cumprod_Q_t = cumprod_Q_t[:,None,:,:]
        self.cumprod_Q_t_1 = cumprod_Q_t_1[:,None,:,:]
        
        self.to_domain = lambda xt: xt.argmax(dim=-1)  
        
        def to_one_hot(xt):
            one_hot = torch.zeros_like(xt).reshape(-1,vocab_size)
            one_hot[torch.arange(one_hot.shape[0]), xt.flatten()] = 1
            return one_hot.reshape(bs,*self.object_shape)
        self.from_domain = lambda xt: to_one_hot(xt)
        pass
    
    
    def torch_dist(self, probs):
        return Categorical(probs)

    
    def forward_step_distribution(self, x0, t):
        return (
            torch.matmul(x0, self.cumprod_Q_t[t]),
        )


    def forward_markov_step_distribution(self, xt, t):
        raise NotImplementedError

    
    def reverse_step_distribution(self, x0, xt, t):
        return (
            torch.matmul(xt, self.Q_t[t].transpose(-1,-2)) * 
            torch.matmul(x0, self.cumprod_Q_t_1[t]) /
            (torch.matmul(x0, self.cumprod_Q_t[t]) * xt).sum(dim=-1)[...,None],
        )


    def result_distribution(self, batch):
        bs, dev = batch['batch_size'], batch['device']
        return ( 
            torch.ones((bs,*self.object_shape), dtype=torch.float32, device=dev)
            / self.object_shape[-1],
        )


    def precompute_xt_normalization_statistics(self, data_generator, num_batches):
        pass


    def time_dependent_xt_normalization(self, xt, t):
        return xt
