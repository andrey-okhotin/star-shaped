import torch

from diffusion.general_diffusion.general_diffusion import GeneralDiffusion




class MarkovDiffusion(GeneralDiffusion):

    """
    DESCRIPTION:
        Main class that describes main features of the classic markovian diffusion model. 
        This class is a parent class for all markovian diffusion models. This class define 
        how to:
            - do train step
            - sample from trained model
            - estimate log-likelihood using IWAE
                **optionally**
            - precompute and use time-dependent xt normalization

        So, you can create your own markovian diffusion model inheriting from this class, 
        than you only need to define:
                **necessary**
            - noise distribution
            - forward process noise schedule
            - forward step distribution q(xt|x0)
            - reverse step distribution q(xt-1 | x_t, x0_prediction)
            - limit distribution (result distribution) p(xT)

                **optionally**
            for correct log-likelihood estimation:
            - forward markov step distribution q(x_t | x_{t-1})
            - dequantization q(x0|x1)

        There are 2 examples of classic markovian models:
            1)  DDPM for image data and SyntheticDDPM for synthetic data: 
                    - lib/diffusion/image_data_diffusion/ddpm_diffusion.py
                    - lib/diffusion/synthetic_data_diffusion/synthetic_ddpm_diffusion.py
                Difference only in using normalization for neural network input, for images
                it's not necessary, but on synthetic data it has an impact.
            2)  D3PM for dicrete data and experiments on Text8
                    - lib/diffusion/text_data_diffusion/d3pm_diffusion.py
    
    """
    
    def __init__(self, diffusion_config):
        super().__init__(diffusion_config)
        pass


    """
    Following 3 methods exist only for markovian diffusion models. They operate
    with distribution q(x_t | x_{t-1}) that doesn't exist in the case of the SS-DDPM.
    """

    
    def forward_markov_step_distribution(self, xt, t):
        raise NotImplementedError




    def forward_markov_step_sample(self, xt, t):
        return self.from_domain(self.sample(*self.forward_markov_step_distribution(xt, t)))
    


    
    def forward_markov_log_prob(self, xt, t, x):
        log_probs = self.torch_dist(*self.forward_markov_step_distribution(xt, t)
                                   ).log_prob(self.to_domain(x))
        bs = x.shape[0]
        return log_probs.reshape(bs,-1).mean(dim=1)

    
    

    def reverse_step_distribution(self, x0, xt, t):
        """
        q(x_{t-1} | x_t, x_0)
        """
        raise NotImplementedError
        


    
    def precompute_xt_normalization_statistics(self, data_generator, num_batches):
        """
        DESCRIPTION:
            Implementation of time dependent xt normalization. This method precompute
            mean and std statistics for random variables 
                xt ~ q(x_t) = EXPECTATION_{q(x_0)} q(x_t|x_0)
            Statistics estimates will be saved in object attributes: self.xt_mean and self.xt_std.
            This method is very useful for experiments on synthetic data. It also demonstrates 
            analogue of "precompute_tail_normalization_statistics" method in SS-DDPM.
            
        INPUT:
            <>  data_generator - object with attribute "train_loader" -> torch.utils.data.Dataloader
                    Objects shapes must be compatible with defined diffusion model.
                    
            <>  num_batches -> int: number of batches for statistics estimation. Finally
                    each statistic will be computed using N samples.
                        N = batch_size * num_batches
            
        """
        N = self.num_steps
        self.xt_mean = torch.zeros((N,), dtype=torch.float32, device=self.device)
        self.xt_std = torch.zeros((N,), dtype=torch.float32, device=self.device)
        
        batch_index = 0
        while batch_index < num_batches - 1:
            for x0 in data_generator.train_loader:
                x0 = data_generator.create_batch(x0)['x0']
                x0 = x0.to(self.device)
                bs = x0.shape[0]
                expand = lambda series: series.flatten().repeat(bs).reshape(bs, N).T.flatten()
                all_time = expand(torch.arange(N, device=self.device))
                xt = self.forward_step_sample(x0.repeat(N,1,1,1), all_time).reshape(N,-1)
                xt_mean, xt_std = xt.mean(dim=1), xt.std(dim=1)
                self.xt_mean += xt_mean / num_batches
                self.xt_std += xt_std / num_batches
                
                batch_index += 1
                if batch_index == num_batches - 1:
                    break
        
        self.xt_mean = self.xt_mean.reshape(-1,1,1,1)
        self.xt_std = self.xt_std.reshape(-1,1,1,1)
        pass


    
    
    def time_dependent_xt_normalization(self, xt, t):
        return (xt - self.xt_mean[t]) / self.xt_std[t]


    
    
    def kl(self, batch):
        """
        DESCRIPTION:
            Calculates KL terms:
                KL[ q(x_{t-1} | x_t, x_0) || q(x_{t-1} | x_{t-1}, x0_prediction) ]
            using torch.distributions.kl.kl_divergence.
            
            OR
            
            This method may be overridden in the particular model.
        
        """
        x0, x0_pred, x_t, t = batch['x0'], batch['x0_prediction'], batch['x_t'], batch['t']
        return kl_divergence(
            self.torch_dist(*self.reverse_step_distribution(x0, x_t, t)),
            self.torch_dist(*self.reverse_step_distribution(x0_pred, xt, t))
        )


    

    def model_prediction(self, model, xt, t):
        """
        DESCRIPTION:
            How to get prediction of the x0 using 
                neural network - model
                xt             - previous state
                t              - time moment
                
        """
        normed_xt = self.time_dependent_xt_normalization(xt, t)
        rescaled_t = self.time_rescaler(t)
        x0_pred = model(normed_xt, rescaled_t)
        return x0_pred


    

    def train_step(self, model, batch, loss_object, mode):
        """
        DESCRIPTION:
            How to do a single train step. On each step we
                1. sample time moments t
                2. sample xt for particular x0
                3. predict x0 using neural network, xt and t
        
        """
        batch['t'] = self.time_distribution.sample(
            batch, 
            loss_object,
            mode
        )
        batch['xt'] = self.forward_step_sample(
            batch['x0'], 
            batch['t']
        )
        batch['x0_prediction'] = self.model_prediction(
            model,
            batch['xt'], 
            batch['t']
        )
        return batch


    

    def sampling_procedure(self, batch, model, progress_printer, num_sampling_steps):
        """
        DESCRIPTION:
            Unified sampling procedure for all markovian diffusion models.
        
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
        for t_vector in reverse_time_iterator:
            batch['t'] = t_vector.to(batch['device'])
            t_value = batch['t'][0].item()
            progress_printer(t_value)
            if t_value in model_estimation_points:
                batch['x0_prediction'] = self.model_prediction(
                    model,
                    batch['xt'], 
                    batch['t']
                )
            batch['xt'] = self.reverse_step_sample(
                batch['x0_prediction'],
                batch['xt'],
                batch['t']
            )
        result_object = batch['xt'].cpu().clone()
        return result_object




    def ll_estimation(self, batch, model):
        """
        DESCRIPTION:
            How to estimate log-likelihood in markovian diffusion modeks. This algorithm needs 
            predefined dequantization procedure: 
                method: self.dequantization(x1, x0_pred, x0, t) -> 
                                                probability q(x_0 | x1, x0_prediction, t)
            Log-likelihood estimates will be saved in batch['ll']
        
        """
        batch['ll'] = torch.zeros(
            (batch['batch_size'],), dtype=batch['x0'].dtype, device=batch['device'])
        forward_time_iterator = self.time_distribution.forward_time_iterator(
            batch, start_from=0
        )

        # t = 1
        batch['t'] = self.time_distribution.get_time_points_vector(batch, t=1)
        batch['xt'] = self.forward_markov_step_sample(
            batch['x0'],
            batch['t']
        )
        batch['x0_prediction'] = self.model_prediction(
            model,
            batch['x1'],
            batch['t']
        )
        log_p = self.diffusion.dequantization(
            batch['x1'],
            batch['x0_prediction'],
            batch['x0']
        )
        log_q = self.reverse_log_prob(
            batch['x0'],
            batch['xt'],
            batch['t'],
            batch['x0']
        )
        batch['ll'] = log_p - log_q
        
        for t_vector in forward_time_iterator:
            batch['t'] = t_vector.to(batch['device'])
            t_index = batch['t'][0].item()
            
            batch['xt'] = self.forward_markov_step_sample(
                batch['xt-1'],
                batch['t']
            )
            batch['x0_prediction'] = self.model_prediction(
                model,
                batch['xt'],
                batch['t']
            )
                
            if t_index == 0:
                batch['log_p_t'] += self.dequantizator.log_prob(
                    batch['x0_prediction'],
                    batch['x0']
                )
            else:
                batch['log_q_t'] += self.reverse_log_prob(
                    batch['x0'],
                    batch['xt'],
                    batch['t'],
                    batch['xt-1']
                )
                batch['log_p_t'] += self.reverse_log_prob(
                    batch['x0_prediction'],
                    batch['xt'],
                    batch['t'],
                    batch['xt-1'],
                    batch['noise_coef']
                )
            batch['xt-1'] = batch['xt']

            if t_index == 1:
                break
            
        batch['log_q_t'] += self.forward_log_prob(
            batch['x0'],
            batch['t'],
            batch['xt']
        )
        batch['log_p_t'] += self.result_distribution_log_prob(
            batch['xt'],
            batch
        ) 

        L = (batch['log_p_t'] - batch['log_q_t']).reshape(
            self.num_iwae_estimations, self.bs
        ).cpu()
        K = torch.tensor(self.num_iwae_estimations)
        L = torch.logsumexp(L, dim=0) - torch.log(K)
        iwae_log_probs = torch.hstack((iwae_log_probs, L))

        pass

