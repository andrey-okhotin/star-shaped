import torch
from torch.distributions.wishart import Wishart

from diffusion.general_diffusion.star_shaped_diffusion import StarShaped
from diffusion.schedulers.nongaussian_schedulers import nongaussian_scheduler




class WishartStarShaped(StarShaped):
    
    """
    DESCRIPTION:
        SS-DDPM with Wishart distribution. Used for experiments on symmetric positive definite
        matrices of size 2x2.
        
    """
    
    def __init__(self, diffusion_config):
        super().__init__(diffusion_config)
        xi_t, n_t, self.time_rescaler = nongaussian_scheduler(
            diffusion_config.scheduler, self.num_steps)
        
        xi_t_1 = torch.hstack((torch.tensor([xi_t[0]]), xi_t[:-1]))
        n_t_1 = torch.hstack((torch.tensor([n_t[0]]), n_t[:-1]))
        
        self.xi_t, self.xi_t_1 = xi_t.view(-1,1,1,1), xi_t_1.view(-1,1,1,1)
        self.n_t, self.n_t_1 = n_t.view(-1,1,1,1), n_t_1.view(-1,1,1,1)
        self.n_T = torch.tensor([self.n_t[-1,0,0,0].item()])
        
        self.at = self.n_t * self.xi_t
        self.T = lambda xt: xt
        self.to_domain = lambda xt: xt
        self.from_domain = lambda xt: xt
        pass
    
    
    def torch_dist(self, nt, Vt):
        return Wishart(df=nt[:,:,0,0], covariance_matrix=Vt)
    

    def forward_step_distribution(self, x0, t):
        bs = x0.shape[0]
        I = create_eye(bs, x0.device, self.object_shape[-1])
        mu_t = self.xi_t[t] * fast_inverse(self.to_domain(x0)) + (1 - self.xi_t[t]) * I
        n_t = self.n_t[t]
        return (
            n_t,
            fast_inverse(mu_t) / n_t
        )
    
    
    def reverse_step_distribution(self, x0, t):
        bs = x0.shape[0]
        I = create_eye(bs, x0.device, self.object_shape[-1])
        mu_t_1 = self.xi_t_1[t] * fast_inverse(self.to_domain(x0)) + (1 - self.xi_t_1[t]) * I
        n_t_1 = self.n_t_1[t]
        return (
            n_t_1,
            fast_inverse(mu_t_1) / n_t_1
        )
    
    
    def result_distribution(self, batch):
        bs = batch['batch_size']
        I = create_eye(bs, batch['device'], self.object_shape[-1])
        df = self.n_T.repeat(bs).reshape(bs,1,1,1)
        return (
            df, 
            I / df
        )
    

    def kl(self, batch):
        """
        DESCRIPTION:
            Implementation of KL divergence between two Wishart distributions for
            pytorch distributions that allows to backpropagate through second argument
            in KL divergence.

        """
        x0, x0_pred, t = batch['x0'], batch['x0_prediction'], batch['t']
        w0 = self.torch_dist(*self.reverse_step_distribution(x0, t))
        w1 = self.torch_dist(*self.reverse_step_distribution(x0_pred, t))
        
        n0, n1 = w0.df.flatten(), w1.df.flatten()
        V0, V1 = w0.covariance_matrix, w1.covariance_matrix
        bs, H = V0.shape[0], V0.shape[-1]

        matmul_inv_V1_V0 = torch.matmul(fast_inverse(V1), V0)
        det_term = (-1) * n1 * torch.log(torch.det(matmul_inv_V1_V0).flatten())
        I = create_eye(bs, V0.device, self.object_shape[-1])
        trace_term = n0 * (matmul_inv_V1_V0 * I).reshape(V0.shape[0], -1).sum(dim=1)
        p = V0.shape[-1]
        gamma_proportion = 2 * (torch.mvlgamma(n1 / 2, p) - torch.mvlgamma(n0 / 2, p))
        difference = (n0 - n1) * mvdigamma(n0 / 2, p)
        shift = (-1) * n0 * p
        kl = 0.5 * (det_term + trace_term + gamma_proportion + difference + shift).flatten()
        return kl


    def kl_rescaled(self, batch):
        """
        DESCRIPTION:
            Reweighted version of ELBO objective for Beta SS-DDPM case. Analogue of
            L_simple in classic DDPM.
        
        """
        return self.kl(batch) / self.n_t[batch['t']]





def mvdigamma(t, p):
    x = t.clone()
    x.requires_grad = True
    f = torch.mvlgamma(x, p).sum()
    f.backward()
    return x.grad




def fast_inverse(mx):
    # implementation for matrices 2x2
    # speed x5 compared to default pytorch~1.09 
    det = mx[...,0,0] * mx[...,1,1] - mx[...,0,1] * mx[...,1,0]
    m = torch.hstack(
        (mx[...,1,1], -mx[...,0,1], -mx[...,1,0], mx[...,0,0])
    ).reshape(-1,1,2,2)
    return m / det[...,None,None]




def create_eye(bs, dev, size):
    return torch.eye(size, device=dev)[None,None,:,:].repeat(bs,1,1,1)