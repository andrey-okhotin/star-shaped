import torch
from torch.distributions.beta import Beta

from diffusion.general_diffusion.star_shaped_diffusion import StarShaped
from diffusion.schedulers.nongaussian_schedulers import nongaussian_scheduler




class BetaStarShaped(StarShaped):
    
    """
    DESCRIPTION:
        SS-DDPM with Beta distribution. Used for experiments on CIAFR10.
        
    """
    
    def __init__(self, diffusion_config):
        super().__init__(diffusion_config)
        nu_t, xi_t, self.time_rescaler = nongaussian_scheduler(
            diffusion_config.scheduler, self.num_steps)
        
        nu_t_1 = torch.hstack((torch.tensor([nu_t[0]+2]), nu_t[:-1]))
        xi_t_1 = torch.hstack((torch.tensor([xi_t[0]/2]), xi_t[:-1]))
        
        self.nu_t, self.xi_t = nu_t.view(-1,1,1,1), xi_t.view(-1,1,1,1)
        self.nu_t_1, self.xi_t_1 = nu_t_1.view(-1,1,1,1), xi_t_1.view(-1,1,1,1)
        
        def MuNu2AlphaBeta(mu, nu):
            return mu * nu, (1 - mu) * nu
        self.MuNu2AlphaBeta = MuNu2AlphaBeta
        
        self.at = self.nu_t * (1 - self.xi_t)        
        self.T = lambda xt: torch.log(xt / (1 - xt))
        self.to_domain = lambda xt: torch.clip((xt + 1) / 2, 1e-4, 1-1e-4)
        self.from_domain = lambda xt: 2 * xt - 1
        pass

    
    def torch_dist(self, alpha, beta):
        return Beta(torch.clip(alpha, min=1e-4), torch.clip(beta, min=1e-4))
    

    def forward_step_distribution(self, x0, t):
        return self.MuNu2AlphaBeta(
            0.5 * self.xi_t[t] + (1 - self.xi_t[t]) * self.to_domain(x0),
            self.nu_t[t]
        )
    
    
    def reverse_step_distribution(self, x0, t):
        return self.MuNu2AlphaBeta(
            0.5 * self.xi_t_1[t] + (1 - self.xi_t_1[t]) * self.to_domain(x0),
            self.nu_t_1[t]
        )
    
    
    def result_distribution(self, batch):
        bs, dev = batch['batch_size'], batch['device']
        return (
            torch.ones((bs,*self.object_shape), dtype=torch.float32, device=dev),
            torch.ones((bs,*self.object_shape), dtype=torch.float32, device=dev),
        )
    
    
    def kl_rescaled(self, batch):
        """
        DESCRIPTION:
            Reweighted version of ELBO objective for Beta SS-DDPM case. Analogue of
            L_simple in classic DDPM.
        
        """
        x0, x0_pred, t = batch['x0'], batch['x0_prediction'], batch['t']
        xi, nu = self.xi_t_1[t], self.nu_t_1[t]
        mu = 0.5 * xi + (1 - xi) * self.to_domain(x0)
        mu_pred = 0.5 * xi + (1 - xi) * self.to_domain(x0_pred)
        term_1 = (mu_pred - mu) * (torch.polygamma(0, nu * (1. - mu)) - torch.polygamma(0, nu * mu))
        term_2 = torch.lgamma(nu * (1. - mu_pred)) - torch.lgamma(nu * (1. - mu))
        term_3 = torch.lgamma(nu * mu_pred) - torch.lgamma(nu * mu)
        return term_1 + term_2 / nu + term_3 / nu


    def model_prediction(self, model, Gt, t):
        """
        DESCRIPTION:
            Compared to the parent SS-DDPM method, we additionally put sigmoid on
            the top of the network to map predictions to the domain of Beta distribution.
            
        """
        normed_Gt = self.time_dependent_tail_normalization(Gt, t)
        rescaled_t = self.time_rescaler(t)
        x0_pred = self.from_domain(
            torch.sigmoid(model(normed_Gt, rescaled_t))
        )
        return x0_pred



