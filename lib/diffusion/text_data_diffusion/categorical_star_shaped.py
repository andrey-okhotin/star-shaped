import torch
from torch.nn.functional import softmax
from torch.distributions.categorical import Categorical

from diffusion.general_diffusion.star_shaped_diffusion import StarShaped
from diffusion.schedulers.nongaussian_schedulers import nongaussian_scheduler




class CategoricalStarShaped(StarShaped):
    
    """
    DESCRIPTION:
        SS-DDPM with Categorical distribution. Used for experiments on Text8.
        
    """
    
    def __init__(self, diffusion_config):
        super().__init__(diffusion_config)
        betas_t, _, cumprod_Q_t, self.time_rescaler = nongaussian_scheduler(
            diffusion_config.scheduler, self.num_steps)
        
        cumprod_Q_t_1 = torch.vstack((
            torch.eye(*cumprod_Q_t[0].shape)[None,...], cumprod_Q_t[:-1]
        ))
        self.betas_t = betas_t
        self.cumprod_Q_t = cumprod_Q_t[:,None,:,:]
        self.cumprod_Q_t_1 = cumprod_Q_t_1[:,None,:,:]
        
        self.T = None
        self.at = None
        
        # mappings TO and FROM onehot representation
        self.to_domain = lambda xt: xt.argmax(dim=-1)  
        vocab_size = self.object_shape[-1]
        def to_one_hot(xt):
            bs, _, seqlen = xt.shape[:3]
            one_hot = torch.zeros((bs*seqlen, vocab_size), dtype=torch.float32, device=xt.device)
            one_hot[torch.arange(bs*seqlen), xt.flatten()] = 1
            return one_hot.reshape(bs,*self.object_shape)
        self.from_domain = lambda xt: to_one_hot(xt)
        pass
    
    
    def torch_dist(self, probs):
        return Categorical(probs=probs)
    
    
    def tail_statistic_term(self, xt, t):
        return torch.log(torch.matmul(xt, self.cumprod_Q_t[t].transpose(-2,-1)))
    

    def forward_step_distribution(self, x0, t):
        return torch.matmul(x0, self.cumprod_Q_t[t]),
    
    
    def reverse_step_distribution(self, x0, t):
        return torch.matmul(x0, self.cumprod_Q_t_1[t]),
    
    
    def result_distribution(self, batch):
        bs, dev = batch['batch_size'], batch['device']
        return ( 
            torch.ones((bs,*self.object_shape), dtype=torch.float32, device=dev)
            / self.object_shape[-1],
        )


    def precompute_tail_normalization_statistics(self, data_generator, num_batches):
        pass
    

    def time_dependent_tail_normalization(self, Gt, t):
        """
        DESCRIPTION:
            Alternative way to normalize Gt in the case, when we work with
            Categorical distribution.
        
        """
        return softmax(Gt, dim=-1)


    def dequantization(self, G2, x0_pred, x0, t):
        """
        DESCRIPTION:
            In the case of Categorical SS-DDPM we have an analitical procedure
            to calculate q(x_0|G_1). You can find its deriviation in the paper.
            
        """
        bs = x0.shape[0]
        x1 = self.reverse_step_sample(x0, t)
        log_p = self.reverse_log_prob(x0_pred, t+1, x1)
        log_q = self.reverse_log_prob(x0, t+1, x1)
        kl = log_q - log_p
        eps = 1e-3 * torch.ones((self.object_shape[-1],), dtype=torch.float32, device=x0.device)
        new_x1 = (x1 + eps.reshape(1,1,1,-1)) / (1 + eps.sum())
        t = torch.zeros_like(x1[:,0,0,0]).to(torch.long)
        G1 = G2 + torch.log(torch.matmul(new_x1, self.cumprod_Q_t_1[t].transpose(-2,-1)))
        dequant = x0 * torch.log(softmax(G1, dim=-1))
        return dequant.reshape(bs,-1).mean(dim=-1) - kl




