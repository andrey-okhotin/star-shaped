import torch
import torch.nn as nn
from torch.distributions.kl import kl_divergence

from diffusion.general_diffusion.general_diffusion import GeneralDiffusion




class StarShaped(GeneralDiffusion):
    
    """
    DESCRIPTION:
        Main class that describes main features of the proposed model. This class is a
        parent class for all SS-DDPM models. Here you can find standard operations with
        our model. This class define how to:
            - sample sufficient tail statistic Gt
            - update Gt during sampling
            - precompute and use time-dependent tail normalization
            - do train step
            - sample from trained model
            - estimate log-likelihood using IWAE

        So, you can create your own SS-DDPM model inheriting from this class, than you
        only need to define:
                **necessary**
            - noise distribution
            - forward process noise schedule
            - forward step distribution q(xt|x0)
            - reverse step distribution q(xt-1|x0_prediction)
            - limit distribution (result distribution) p(xT)

                **optionally**
            - dequantization q(x0|G1) for correct log-likelihood estimation

        There are 5 examples of SS-DDPM models:
            1)  Dirichlet SS-DDPM for 2-dimensional simplex data: 
                    - lib/diffusion/synthetic_data_diffusion/dirichlet_star_shaped.py
            2)  Wishart SS-DDPM for symmetric positive-definite matrices 2x2
                    - lib/diffusion/synthetic_data_diffusion/wishart_star_shaped.py
            3)  von Mises-Fisher SS-DDPM for data on 2-dimensional sphere
                    - lib/diffusion/geodesic_data_diffusion/von_mises_fisher_star_shaped.py
            4)  Beta SS-DDPM for data on [0,1] segment (applicable to images)
                    - lib/diffusion/image_data_diffusion/beta_star_shaped.py
            5)  Categorical SS-DDPM for discrete data (applicable to texts)
                    - lib/diffusion/text_data_diffusion/categorical_star_shaped.py
    
    """
    
    def __init__(self, diffusion_config):
        super().__init__(diffusion_config)
        def reverse_cumsum(series):  # reverse -> cumsum -> reverse
            return torch.flip(torch.cumsum(torch.flip(series, dims=[0]), dim=0), dims=[0])
        self.reverse_cumsum = reverse_cumsum
        pass


    
    
    def reverse_step_distribution(self, x0, t):
        """
        q(x_{t-1} | x_0)
        """
        raise NotImplementedError


    

    def tail_statistic_term(self, xt, t):
        """
        a_t * T(x_t)
        """
        return self.at[t] * self.T(self.to_domain(xt))


    
    
    def sample_Gt(self, x0, t):
        """
        x_s ~ q(x_s|x_0) for s = t,T
        G_t = SUM_{s>t} a_s * T(x_s)
        """
        N = self.num_steps
        bs = x0.shape[0]
        expand = lambda series: series.flatten().repeat(bs).reshape(bs, N).T.flatten()
        all_time = expand(torch.arange(N, device=self.device))
        xs = self.forward_step_sample(x0.repeat(N,1,1,1), all_time)
        Gt = self.tail_statistic_term(xs, all_time).reshape(N,bs,*self.object_shape)
        Gt_idx = (all_time.reshape(N, bs) >= t).reshape(N,bs,1,1,1)
        Gt = torch.sum(Gt * Gt_idx, 0)
        return Gt


    
    
    def update_Gt(self, xt, Gt, t):
        """
        G_{t-1} = G_t + a_t * T(x_t)
        """
        return Gt + self.tail_statistic_term(xt, t)


    
    
    def precompute_tail_normalization_statistics(self, data_generator, num_batches):
        """
        DESCRIPTION:
            Implementation of time dependent tail normalization. This method precompute
            mean and std statistics for random variables 
                Gt ~ q(G_t) = EXPECTATION_{q(x_0)} q(G_t|x_0)
            Statistics estimates will be saved in object attributes: self.Gt_mean and self.Gt_std
            
        INPUT:
            <>  data_generator - object with attribute "train_loader" -> torch.utils.data.Dataloader
                    Objects shapes must be compatible with defined diffusion model.
                    
            <>  num_batches -> int: number of batches for statistics estimation. Finally
                    each statistic will be computed using N samples.
                        N = batch_size * num_batches
            
        """
        N = self.num_steps
        self.Gt_mean = torch.zeros((N), dtype=torch.float32, device=self.device)
        self.Gt_std = torch.zeros((N), dtype=torch.float32, device=self.device)

        batch_index = 0
        while batch_index < num_batches - 1:
            for x0 in data_generator.train_loader:
                x0 = data_generator.create_batch(x0)['x0']
                x0 = x0.to(self.device)
                bs = x0.shape[0]
                expand = lambda series: series.flatten().repeat(bs).reshape(bs,N).T.flatten()
                all_time = expand(torch.arange(N, device=x0.device))
                xs = self.forward_step_sample(x0.repeat(N,1,1,1), all_time)
                Gt_terms = self.tail_statistic_term(xs, all_time).reshape(N,-1)
                Gt = self.reverse_cumsum(Gt_terms)
                Gt_mean, Gt_std = Gt.mean(dim=1), Gt.std(dim=1)
                self.Gt_mean += Gt_mean / num_batches
                self.Gt_std += Gt_std / num_batches
                
                batch_index += 1
                if batch_index == num_batches - 1:
                    break
        
        self.Gt_mean = self.Gt_mean.reshape(N,1,1,1)
        self.Gt_std = self.Gt_std.reshape(N,1,1,1)
        pass


    
    
    def time_dependent_tail_normalization(self, Gt, t):
        return (Gt - self.Gt_mean[t]) / self.Gt_std[t]
    


    
    def kl(self, batch):
        """
        DESCRIPTION:
            Calculates KL terms:
                KL[ q(x_{t-1} | x_0) || q(x_{t-1} | x0_prediction) ]
            using torch.distributions.kl.kl_divergence
        
            OR
            
            This method may be overridden in the particular model.
            
        """
        x0, x0_pred, t = batch['x0'], batch['x0_prediction'], batch['t']
        return kl_divergence(
            self.torch_dist(*self.reverse_step_distribution(x0, t)),
            self.torch_dist(*self.reverse_step_distribution(x0_pred, t))
        )


    
    
    def model_prediction(self, model, Gt, t):
        """
        DESCRIPTION:
            How to get prediction of the x0 using 
                neural network - model
                Gt             - sufficient tail statistic
                t              - time moment
                
        """
        normed_Gt = self.time_dependent_tail_normalization(Gt, t)
        rescaled_t = self.time_rescaler(t)
        x0_pred = model(normed_Gt, rescaled_t)
        return x0_pred


    

    def train_step(self, model, batch, loss_object, mode):
        """
        DESCRIPTION:
            How to do a single train step. On each step we
                1. sample time moments t
                2. sample sufficient tail statistics Gt for particular x0
                3. predict x0 using neural network, Gt and t
        
        """
        batch['t'] = self.time_distribution.sample(
            batch, 
            loss_object,
            mode
        )
        batch['Gt'] = self.sample_Gt(
            batch['x0'], 
            batch['t']
        )
        batch['x0_prediction'] = self.model_prediction(
            model,
            batch['Gt'], 
            batch['t']
        )
        return batch


    

    def sampling_procedure(self, batch, model, progress_printer, num_sampling_steps):
        """
        DESCRIPTION:
            Unified sampling procedure for all Star-Shaped DDPMs. Pseudocode of this 
            algorithm you can find in the paper.
        
        """
        # affects sampling only if you want to do fewer sampling steps than you have 
        # in your discretization: select time points where we use model to predict x0
        if num_sampling_steps == -1:
            num_sampling_steps = self.num_steps
        num_steps = self.num_steps
        model_estimation_points = [ 0 ]
        for t in range(num_sampling_steps - 1):
            step = int(num_steps / (num_sampling_steps - 1 - t))
            model_estimation_points.append(model_estimation_points[-1] + step)
            num_steps -= step
        model_estimation_points[-1] = self.num_steps - 1
    
        # samlping procedure
        batch['xt'] = self.result_distribution_sample(batch)
        reverse_time_iterator = self.time_distribution.reverse_time_iterator(
            batch, start_from=self.num_steps-1
        )
        batch['Gt'] = torch.zeros_like(batch['xt'])
        for t_vector in reverse_time_iterator:
            batch['t'] = t_vector.to(batch['device'])
            t_value = batch['t'][0].item()
            progress_printer(t_value)
            batch['Gt'] = self.update_Gt(
                batch['xt'],
                batch['Gt'],
                batch['t']
            )
            if t_value in model_estimation_points:
                batch['x0_prediction'] = self.model_prediction(
                    model,
                    batch['Gt'], 
                    batch['t']
                )
            batch['xt'] = self.reverse_step_sample(
                batch['x0_prediction'],
                batch['t']
            )
        result_object = batch['x0_prediction'].cpu().clone()
        return result_object


    
    
    def ll_estimation(self, batch, model):
        """
        DESCRIPTION:
            How to estimate log-likelihood in SS-DDPM. This code used only to get the results
            of Categorical SS-DDPM on Text8. This algorithm needs predefined dequantization 
            procedure: 
                method: self.dequantization(G2, x0_pred, x0, t) -> 
                                                probability q(x_0 | G_2, x0_prediction, t)
            Log-likelihood estimates will be saved in batch['ll']
        
        """
        batch['ll'] = torch.zeros(
            (batch['batch_size'],), dtype=batch['x0'].dtype, device=batch['device'])
        
        # t = T
        batch['t'] = self.time_distribution.get_time_points_tensor(
            batch, t=-1
        )
        batch['xt'] = self.forward_step_sample(
            batch['x0'],
            batch['t']
        )
        log_p = self.result_distribution_log_prob(
            batch['xt'],
            batch
        )
        log_q = self.forward_log_prob(
            batch['x0'],
            batch['t'],
            batch['xt']
        )
        kl = log_q - log_p
        batch['ll'] -= kl
    
        # t = T-1, ..., 2
        reverse_time_iterator = self.time_distribution.reverse_time_iterator(
            batch, start_from=self.num_steps - 2
        )
        batch['Gt+1'] = torch.zeros_like(batch['xt'])
        for t_vector in reverse_time_iterator:
            batch['t'] = t_vector.to(batch['device'])
            batch['t+1'] = batch['t'] + 1
            t_index = batch['t'][0].item()
    
            batch['xt+1'] = batch['xt']
            batch['Gt+1'] = self.update_Gt(
                batch['xt+1'],
                batch['Gt+1'],
                batch['t+1']
            )
            batch['x0_prediction'] = self.model_prediction(
                model,
                batch['Gt+1'], 
                batch['t+1']
            )
            batch['xt'] = self.forward_step_sample(
                batch['x0'],
                batch['t']
            )
            log_p = self.reverse_log_prob(
                batch['x0_prediction'],
                batch['t+1'],
                batch['xt']
            )
            log_q = self.reverse_log_prob(
                batch['x0'],
                batch['t+1'],
                batch['xt']
            )
            kl = log_q - log_p
            batch['ll'] -= kl
    
        batch['xt+1'], batch['t+1'] = batch['xt'], batch['t']
        
        # t = 1
        batch['Gt+1'] = self.update_Gt(
            batch['xt+1'],
            batch['Gt+1'],
            batch['t+1']
        )
        batch['x0_prediction'] = self.model_prediction(
            model,
            batch['Gt+1'], 
            batch['t+1']
        )
        batch['ll'] += self.dequantization(
            batch['Gt+1'],
            batch['x0_prediction'],
            batch['x0'],
            batch['t']
        )
        return batch['ll'].cpu()
    
    
    """
    Following methods used only for log-likelihood estimation. Particular exmaples
    of these methods you can find for Categorical SS-DDPM in file:
        - lib/diffusion/text_data_diffusion/categorical_star_shaped.py
    """
    
    
    def set_dequantizator(self, dequantizator_model):
        self.dequantizator_model = dequantizator_model
    


    
    def dequantization(self, G2, x0_pred, x0, t):
        return self.dequantizator_model(x0_pred).log_prob(x0)


