import torch
import random
from IPython.display import clear_output

from data.synthetic_datasets.UniversalDataset import init_universal_dataset
from data.synthetic_datasets.DataloaderWithNoiseSampler import init_fast_dataloaders
from models.Models import init_model
from diffusion.Diffusion import init_diffusion
from metrics_and_losses.SimpleMetrics import init_simple_metric
from metrics_and_losses.Loss import LossObject
from optimizers.Optimizers import Optimizer
from saving_utils.Checkpoints import Checkpoints




class SimpleTrainer:
    
    def __init__(self, config):
        self.config = config
        self.dataset = init_universal_dataset(config.dataset_config)
        self.model = init_model(config.model_config)
        self.diffusion = init_diffusion(config.diffusion_config)
        self.loss = LossObject(config.loss_config)
        self.ema_val_loss = LossObject(config.loss_config)
        self.optimizer = Optimizer(self.model, self.config.optimizer_config)
        self.metrics = []
        for metric_config in config.metrics_config:
            self.metrics.append(init_simple_metric(metric_config))
        if not (config.checkpoints_config is None):
            self.model_checkpoints = Checkpoints(config.checkpoints_config, 0)
        
        self.norm_samples = config.norm_samples
        self.ema_val_freq = config.ema_val_freq
        self.show_freq = config.show_freq
        pass
        
        
        
    def train(
        self, 
        method,
        num_iters=350000,
        ema_val_max_iters=100,
        device='cpu',
        num_workers=1,
        show=True,
        graphics_path=None
    ):
        TRAIN_LOADER, _ = init_fast_dataloaders(
            self.dataset,
            num_iters,
            self.diffusion,
            num_workers,
            method
        )
        self.device = device
        self.num_iters = num_iters
        self.model.to(device)
        self.model.device = device
        self.diffusion.to(device)

        # precompute normalization
        if   method == 'ss':
            self.diffusion.precompute_tail_normalization_statistics(
                self.dataset, 
                self.norm_samples
            )
        elif method == 'ddpm':
            self.diffusion.precompute_xt_normalization_statistics(
                self.dataset, 
                self.norm_samples
            )

        # train loop
        try:
            self.train_loss_list, self.ema_val_loss_list = [], []
            self.ema_val_iters = []

            batch = {
                'x0' : None,
                'x0_prediction' : None,
                'xt' : None,
                'Gt' : None,
                't' : None,
                'batch_size' : self.dataset.batch_size,
                'device' : 'cpu'
            }
            self.model.train()
            for i, batch in enumerate(TRAIN_LOADER):
                self.dataset.batch_to(batch, device)

                try:
                    self.diffusion.train_step(
                        batch=batch,
                        model=self.model,
                        loss_object=self.loss,
                        mode='train'
                    )
                    train_loss = self.loss.compute_loss(batch, self.diffusion, self, mode='train')
                    self.optimizer.optimizing_step(train_loss, i)
                    self.train_loss_list.append(train_loss.item())

                    # validation part: check quality of EMA weights
                    if (i+1) % self.ema_val_freq == 0:
                        ema_val_losses = []
                        self.optimizer.switch_to_ema()
                        for j, batch in enumerate(TRAIN_LOADER):
                            self.dataset.batch_to(batch, device)
                            self.diffusion.train_step(
                                batch=batch,
                                model=self.model,
                                loss_object=self.ema_val_loss,
                                mode='validation'
                            )
                            ema_val_losses.append(self.ema_val_loss.compute_loss(
                                batch, self.diffusion, self, mode='validation').item())
                            if j >= ema_val_max_iters:
                                break
                        self.optimizer.switch_from_ema()
                        self.ema_val_loss_list.append(sum(ema_val_losses) / len(ema_val_losses))
                        self.ema_val_iters.append(i)
                except ValueError:
                    pass

                # metrics
                for metric in self.metrics:
                    metric(self, i)

                # checkpointer
                self.model_checkpoints.create_checkpoint(
                    self.model, 
                    self.optimizer, 
                    episode=0, 
                    epoch=i
                )

                # visualization
                if show and (i+1) % self.show_freq == 0:
                    clear_output(wait=True)
                    self.train_mean_loss = self.loss.loss_obj.metric_val.mean(dim=1)
                    self.ema_val_mean_loss = self.ema_val_loss.loss_obj.metric_val.mean(dim=1)
                    for metric in self.metrics:
                        metric.plot(self, graphics_path)

        except KeyboardInterrupt:
            pass

        self.model.cpu()
        self.model.device = 'cpu'
        self.diffusion.to('cpu')
        pass
    
    
    
    def load(self, method):
        if   method == 'ss':
            self.diffusion.precompute_tail_normalization_statistics(
                self.dataset, 
                self.norm_samples
            )
        elif method == 'ddpm':
            self.diffusion.precompute_xt_normalization_statistics(
                self.dataset, 
                self.norm_samples
            )
        pass