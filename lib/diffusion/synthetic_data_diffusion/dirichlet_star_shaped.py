import torch
from torch.nn.functional import softmax
from torch.distributions.dirichlet import Dirichlet

from diffusion.general_diffusion.star_shaped_diffusion import StarShaped
from diffusion.schedulers.nongaussian_schedulers import nongaussian_scheduler




class DirichletStarShaped(StarShaped):
    
    """
    DESCRIPTION:
        SS-DDPM with Dirichlet distribution. Used for experiments on 2d simplex.
        
    """
    
    def __init__(self, diffusion_config):
        super().__init__(diffusion_config)
        nu_t, self.time_rescaler = nongaussian_scheduler(
            diffusion_config.scheduler, self.num_steps)
        
        nu_t_1 = torch.hstack((torch.tensor([nu_t[0]]), nu_t[:-1]))
        self.nu_t, self.nu_t_1 = nu_t.view(-1,1,1,1), nu_t_1.view(-1,1,1,1)
        
        self.at = self.nu_t
        self.T = lambda xt: torch.log(xt)
        self.to_domain = lambda xt: xt
        self.from_domain = lambda xt: xt
        pass
    
    
    def torch_dist(self, alpha):
        return Dirichlet(torch.clip(alpha, min=1e-4))
    

    def forward_step_distribution(self, x0, t):
        return 1 + self.to_domain(x0) * self.nu_t[t],
    
    
    def reverse_step_distribution(self, x0, t):
        return 1 + self.to_domain(x0) * self.nu_t_1[t],
    
    
    def result_distribution(self, batch):
        bs, dev = batch['batch_size'], batch['device']
        return torch.ones((bs,*self.object_shape), dtype=torch.float32, device=dev),




