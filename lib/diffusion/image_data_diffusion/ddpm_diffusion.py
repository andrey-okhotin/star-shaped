import torch
from torch.distributions.normal import Normal

from diffusion.general_diffusion.markov_diffusion import MarkovDiffusion
from diffusion.schedulers.gaussian_schedulers import gaussian_scheduler
    


class DDPM(MarkovDiffusion):
    
    """
    DESCRIPTION:
        Markovian diffusion model with Normal distribution. Used for experiments on 
        CIFAR10. 
        
    """

    def __init__(self, diffusion_config):
        super().__init__(diffusion_config)
        self.config = diffusion_config
        cumprod_alphas_t, self.time_rescaler = gaussian_scheduler(
            diffusion_config.scheduler, self.num_steps)
        
        first_cumprod_alpha_t = cumprod_alphas_t[0]
        eps = 1e-6
        cumprod_alphas_t_1 = torch.hstack((torch.tensor([1.0-eps]), cumprod_alphas_t[:-1]))
        alphas_t = cumprod_alphas_t / cumprod_alphas_t_1
        betas_t = 1 - alphas_t
        
        self.betas_t            = betas_t            = betas_t.view(-1,1,1,1)
        self.alphas_t           = alphas_t           = alphas_t.view(-1,1,1,1)
        self.cumprod_alphas_t   = cumprod_alphas_t   = cumprod_alphas_t.view(-1,1,1,1)
        self.cumprod_alphas_t_1 = cumprod_alphas_t_1 = cumprod_alphas_t_1.view(-1,1,1,1)
        
        self.forward_mean_coef   =    torch.sqrt(cumprod_alphas_t)
        self.forward_std_coef    =    torch.sqrt(1 - cumprod_alphas_t)
        
        self.reverse_mean_coef_1 = (  torch.sqrt(cumprod_alphas_t_1) * betas_t / 
                                      (1 - cumprod_alphas_t)  )
        self.reverse_mean_coef_2 = (  torch.sqrt(alphas_t) * (1 - cumprod_alphas_t_1) /
                                      (1 - cumprod_alphas_t)  )
        self.reverse_std_coef    = (  torch.sqrt(betas_t * (1 - cumprod_alphas_t_1) / 
                                                 (1 - cumprod_alphas_t))  )
        
        self.noise_to_x_coef_1   =    torch.sqrt(1 / cumprod_alphas_t)
        self.noise_to_x_coef_2   =    torch.sqrt(1 / cumprod_alphas_t - 1)
        
        # L_simple - coefficient for rescaling ELBO
        self.l_simple_coefs = ( betas_t**2 / ( 2 * self.reverse_std_coef**2 * alphas_t *
                                          (1 - cumprod_alphas_t) ) )
        self.l_simple_coefs[0,0,0,0]  = self.l_simple_coefs[1,0,0,0]
        
        self.to_domain = lambda xt: xt
        self.from_domain = lambda xt: xt
        pass
    
    
    def torch_dist(self, mu, sigma):
        return Normal(mu, sigma)


    def forward_step_distribution(self, x0, t):
        return (
            self.forward_mean_coef[t] * x0, 
            self.forward_std_coef[t] * torch.ones_like(x0)
        )


    def forward_markov_step_distribution(self, xt, t):
        return (
            torch.sqrt(1 - self.betas_t[t]) * xt,
            torch.sqrt(self.betas_t[t]) * torch.ones_like(xt) 
        )


    def reverse_step_distribution(self, x0, xt, t):
        return (
            self.reverse_mean_coef_1[t] * x0 + self.reverse_mean_coef_2[t] * xt,
            self.reverse_std_coef[t] * torch.ones_like(x0)
        )
    
    
    def result_distribution(self, batch):
        bs, dev = batch['batch_size'], batch['device']
        return (
            torch.zeros((bs,*self.object_shape), dtype=torch.float32, device=dev),
            torch.ones((bs,*self.object_shape), dtype=torch.float32, device=dev),
        )
    
    
    def prediction_reparametrization(self, prediction, xt, t, mode):
        """
        DESCRIPTION:
            How reparametrize neural network predictions for
                - noise prediction
                - x0 prediction
        
        """
        if mode == 'noise2object':
            et = prediction
            x0 = self.noise_to_x_coef_1[t] * xt - self.noise_to_x_coef_2[t] * et
            output = x0
        elif mode == 'object2noise':
            x0 = prediction
            et = (self.noise_to_x_coef_1[t] * xt - x0) / self.noise_to_x_coef_2[t]
            output = et
        return output


    def kl_rescaled(self, batch):
        """
        DESCRIPTION:
            Implementation of L_simple - MSE between added and predicted noises.
            
        """
        if not ('noise_prediction' in batch.keys()):
            batch['noise_prediction'] = diffusion.prediction_reparametrization(
                batch['x0_prediction'],
                batch['xt'],
                batch['t'],
                mode='object2noise'
            )
        if not ('noise' in batch.keys()):
            batch['noise'] = diffusion.prediction_reparametrization(
                batch['x0'],
                batch['xt'],
                batch['t'],
                mode='object2noise'
            )
        bs = batch['batch_size']
        l_simple = ((batch['noise_prediction'] - batch['noise'])**2).reshape(bs, -1).mean(dim=1)
        return l_simple


    def precompute_xt_normalization_statistics(self, data_generator, num_batches):
        if self.config.use_norm:
            super().precompute_xt_normalization_statistics(data_generator, num_batches)
        pass
    

    def time_dependent_xt_normalization(self, xt, t):
        if self.config.use_norm:
            return super().time_dependent_xt_normalization(xt, t)
        return xt


    def model_prediction(self, model, xt, t):
        """
        DESCRIPTION:
            Construct prediction of x0 from noise prediction and previous state xt.
        
        """
        normed_xt = self.time_dependent_xt_normalization(xt, t)
        rescaled_t = self.time_rescaler(t)
        et = model(normed_xt, rescaled_t)
        x0_pred = self.prediction_reparametrization(et, xt, t, mode='noise2object')
        return x0_pred

