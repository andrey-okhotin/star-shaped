import torch
from scipy.special import ive
from tensorflow_probability.substrates.numpy.distributions import VonMisesFisher as VMF

from diffusion.general_diffusion.star_shaped_diffusion import StarShaped
from diffusion.schedulers.nongaussian_schedulers import nongaussian_scheduler




class VonMisesFisherStarShaped(StarShaped):
    
    """
    DESCRIPTION:
        SS-DDPM with von Mises-Fisher distribution. Used for experiments on Geodesic data.
        
    """
    
    def __init__(self, diffusion_config):
        super().__init__(diffusion_config)
        kappa_t, self.time_rescaler = nongaussian_scheduler(
            diffusion_config.scheduler, self.num_steps)
        kappa_t_1 = torch.hstack((torch.tensor([1.1*kappa_t[0]]), kappa_t[:-1]))
        
        self.kappa_t = kappa_t.reshape(-1,1,1,1)
        self.kappa_t_1 = kappa_t_1.reshape(-1,1,1,1)
        
        Ive_1_5 = ive(1.5 + torch.zeros_like(kappa_t), kappa_t)
        Ive_0_5 = ive(0.5 + torch.zeros_like(kappa_t), kappa_t)
        self.kl_coef = (kappa_t * Ive_1_5 / Ive_0_5).reshape(-1,1,1,1) + 1e-4

        self.at = self.kappa_t
        self.T = lambda xt: xt
        self.to_domain = lambda xt: xt
        self.from_domain = lambda xt: xt
        pass
    
    
    def torch_dist(self, mu_t, kappa_t):
        return VonMisesFisher(mu_t, kappa_t)
    

    def forward_step_distribution(self, x0, t):
        return (
            x0,
            self.kappa_t[t]
        )
    
    
    def reverse_step_distribution(self, x0, t):
        return (
            x0,
            self.kappa_t_1[t]
        )
    
    
    def result_distribution(self, batch):
        bs, dev = batch['batch_size'], batch['device']
        mu = torch.zeros((bs,*self.object_shape), dtype=torch.float32, device=dev)
        mu[...,0] += 1
        t = torch.tensor([ self.num_steps-1 ], device=dev).repeat(bs)
        return (
            mu,
            self.kappa_t[t]
        )


    def precompute_tail_normalization_statistics(self, data_generator, num_batches):
        pass
    

    def time_dependent_tail_normalization(self, Gt, t):
        """
        DESCRIPTION:
            Alternative way to normalize Gt in the case, when we work with
            points of sphere.
        
        """
        return Gt / torch.norm(Gt, dim=-1, keepdim=True)
    

    def kl_rescaled(self, batch):
        """
        DESCRIPTION:
            Reweighted version of ELBO objective for Beta SS-DDPM case. Analogue of
            L_simple in classic DDPM.
        
        """
        x0, x0_pred = batch['x0'], batch['x0_prediction']
        return 1. - (x0 * x0_pred).sum(dim=-1)

    
    def kl(self, batch):
        x0, x0_pred, t = batch['x0'], batch['x0_prediction'], batch['t']
        return self.kl_coef[t] * (x0 * (x0 - x0_pred)).sum(dim=-1)




class VonMisesFisher:
    
    """
    DESCRIPTION:
        Class of von Mises Fisher distribution. Use tensorflow_probability VMF
        implementation and wrapped it to allow to work with it in pytorch manner.
        
    """

    def __init__(self, mu, kappa):
        self.mu = mu
        self.kappa = kappa.flatten()
        self.device = mu.device
        pass

    def sample(self):
        vmf = VMF(
            self.mu.detach().cpu().numpy().reshape(self.mu.shape[0],self.mu.shape[-1]), 
            self.kappa.detach().cpu().numpy()
        )
        samples = torch.tensor(vmf.sample(), device=self.device, dtype=torch.float32)
        return samples[:,None,None,:]

    def log_prob(self, x):
        log_C3kappa = (
            torch.log(self.kappa / (2 * torch.pi))
            - self.kappa - torch.log(1 - torch.exp(-2 * self.kappa))
        )
        return log_C3kappa + self.kappa * (self.mu * x).sum(dim=-1).flatten()


    
