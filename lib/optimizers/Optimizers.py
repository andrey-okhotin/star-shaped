import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.distributed as dist

import numpy as np


class Optimizer:
    
    def __init__(self, model, optimization_config):
        """
        """
        self.model = model
        
        optim_method = {
            'Adam' : optim.Adam,
            'SGD' : optim.SGD
        }
        self.optimizer = optim_method[optimization_config.optimizer['method']](
            self.model.parameters(), **optimization_config.optimizer['config'])

        if 'schedulers' in optimization_config:
            scheduler_method = {
                'Exponential'    : lr_scheduler.ExponentialLR,
                'Linear'         : lr_scheduler.LinearLR,
                'Multiplicative' : lr_scheduler.MultiplicativeLR,
                'Lambda'         : lr_scheduler.LambdaLR
            }
            self.schedulers = []
            for scheduler_type, scheduler_config, iters_bounds in optimization_config.schedulers:
                if scheduler_type == 'Lambda':
                    self.schedulers.append((
                        iters_bounds,
                        scheduler_method[scheduler_type](
                            self.optimizer, 
                            lr_lambda = lambda i: 100 / np.sqrt(i+10000)
                        )
                    ))
                else: 
                    self.schedulers.append((
                        iters_bounds,
                        scheduler_method[scheduler_type](self.optimizer, **scheduler_config)
                    ))
                
        if 'clip_grad_norm' in optimization_config:
            self.norm = optimization_config.clip_grad_norm
        
        if 'ema' in optimization_config:
            ema_config = optimization_config.ema
            self.ema_rate = ema_config['ema_rate']
            self.initial_acceleration = 0 if ema_config['initial_acceleration'] else None
            self.ema_parameters = [ 
                p.clone().detach() for p in self.model.parameters() if p.requires_grad
            ]
        pass
    
    
    def zero_grad(self):
        self.optimizer.zero_grad()
    
    
    def optimizing_step(self, loss, iteration=None):
        self.optimizer.zero_grad()
        loss.backward()
        
        # clip gradients
        if hasattr(self, 'norm'):
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                max_norm=self.norm
            )
        
        self.optimizer.step()
        
        # use EMA and control EMA updates
        if hasattr(self, 'ema_rate'):
            ema_rate = self.ema_rate
            # acceleration ema_rate: [ 0.18, 0.25, 0.30, ..., 0.9999, 0.9999, ... ]
            # until dynamical ema_rate rich initial ema_rate
            if not (self.initial_acceleration is None):
                self.initial_acceleration += 1
                ema_rate = min(
                    ema_rate, 
                    (1 + self.initial_acceleration) / (10 + self.initial_acceleration)
                )
            # update ema params using new model params
            with torch.no_grad():
                model_parameters = [p for p in self.model.parameters() if p.requires_grad]
                for ema_param, model_param in zip(self.ema_parameters, model_parameters):
                    ema_param.to(self.model.device)
                    ema_param.sub_((1 - ema_rate) * (ema_param - model_param))
        
        # control lr
        if hasattr(self, 'schedulers'):
            for iters_bounds, scheduler in self.schedulers:
                if iters_bounds[0] <= iteration < iters_bounds[1]: 
                    scheduler.step()
        pass
    
    
    def switch_to_ema(self):
        if hasattr(self, 'ema_rate'):
            self.model_parameters_copy = [ p.clone().cpu().detach()
                for p in self.model.parameters() if p.requires_grad ]
            model_parameters = [p for p in self.model.parameters() if p.requires_grad]
            for ema_param, model_param in zip(self.ema_parameters, model_parameters):
                ema_param.to(self.model.device)
                model_param.data.copy_(ema_param.data)
                ema_param.cpu()
        pass
    
    
    def switch_from_ema(self):
        if hasattr(self, 'ema_rate'):
            model_parameters = [p for p in self.model.parameters() if p.requires_grad]
            for copy_param, model_param in zip(self.model_parameters_copy, model_parameters):
                copy_param.to(self.model.device)
                model_param.data.copy_(copy_param.data)
                copy_param.cpu()
        pass

