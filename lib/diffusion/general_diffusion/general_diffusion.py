import torch
import torch.nn as nn
from torch.distributions.kl import kl_divergence

from diffusion.general_diffusion.time_distribution import TimeDistribution




class GeneralDiffusion:
    
    """
    DESCRIPTION:
        General class for all markovian and non-markovian diffusion models. 
        
        Some methods take 'batch' as a pack of arguments:
            <>  batch -> dict = {
                        **standard fields**
                    'x0'         : x0,
                    'batch_size' : x0.shape[0],
                    'device'     : x0.device

                        **other possible fields**
                    't', 'xt', 'xt+1' 'Gt', 'Gt+1', 'x0_prediction', 'll'

                        **additional info about objects**
                    'resolution', 'channels'
                }
            
    """

    def __init__(self, diffusion_config):
        self.object_shape = diffusion_config.object_shape
        self.num_steps = diffusion_config.num_steps
        self.time_distribution = TimeDistribution(diffusion_config.num_steps)
        self.device = 'cpu'
        pass
    
    
    def torch_dist(self, *args):
        # torch distribution for sampling and KL divergence computation
        raise NotImplementedError
    
    
    def sample(self, *args):
        return self.torch_dist(*args).sample()

    """
    Marginals q(xt|x0) in diffusion model. Similar for markovian and non-markovian
    diffusion models.
    """

    def forward_step_distribution(self, x0, t):
        raise NotImplementedError
    
    
    def forward_step_sample(self, x0, t):
        return self.from_domain(self.sample(*self.forward_step_distribution(x0, t)))
    

    def forward_log_prob(self, x0, t, x):
        log_probs = self.torch_dist(*self.forward_step_distribution(x0, t)
                                   ).log_prob(self.to_domain(x))
        bs = x.shape[0]
        return log_probs.reshape(bs,-1).mean(dim=1)
    
    """
    Reverse process of diffusion model. 
        - markovian reverse steps take arguments: x0, xt, t
        - SS-DDPM reverse steps take only: x0, t
    """
    
    def reverse_step_distribution(self, *args):
        raise NotImplementedError
    
    
    def reverse_step_sample(self, *args):
        return self.from_domain(self.sample(*self.reverse_step_distribution(*args)))


    def reverse_log_prob(self, *args):
        x = args[-1]
        log_probs = self.torch_dist(*self.reverse_step_distribution(*args[:-1])
                                   ).log_prob(self.to_domain(x))
        bs = x.shape[0]
        return log_probs.reshape(bs,-1).mean(dim=1)
    
    """
    Limit disrtibution p(xT) of diffusion model.
    """

    def result_distribution(self, batch):
        raise NotImplementedError


    def result_distribution_sample(self, batch):
        return self.from_domain(self.sample(*self.result_distribution(batch)))


    def result_distribution_log_prob(self, x, batch):
        log_probs = self.torch_dist(*self.result_distribution(batch)
                                   ).log_prob(self.to_domain(x))
        bs = x.shape[0]
        return log_probs.reshape(bs,-1).mean(dim=1)
    
    """
    How to compute loss terms.
        - kl - for ELBO objective
        - kl_rescaled - for reweighted ELBO objective such as L_simple
    """
    
    def kl(self, batch):
        raise NotImplementedError


    def kl_rescaled(self, batch):
        raise NotImplementedError
        
    """
    Device utils.
    """
    
    def to(self, device):
        for atribute in self.__dict__.keys():
            if isinstance(self.__dict__[atribute], torch.Tensor):
                self.__dict__[atribute] = self.__dict__[atribute].to(device)
        self.time_rescaler.discretization = self.time_rescaler.discretization.to(device)
        self.device = device
        pass

    
    def cuda(self, device):
        self.to(device)
        pass
    
    
    def cpu(self):
        self.to('cpu')
        pass
        