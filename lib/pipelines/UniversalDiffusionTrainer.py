import torch
import torch.distributed as dist

from visualization_utils.ProgressPrinters import trainer_printer
from visualization_utils.Plots import plot_loss, plot_metrics
from models.Models import to_distributed

from data.Datasets import init_dataset
from diffusion.Diffusion import init_diffusion
from models.Models import init_model
from metrics_and_losses.Loss import LossObject
from optimizers.Optimizers import Optimizer
from metrics_and_losses.Metrics import MetricsTracker
from saving_utils.Logger import Logger
from saving_utils.Checkpoints import Checkpoints




class UniversalDiffusionTrainer:

    def __init__(self, config, rank):
        self.config = config
        self.data_generator = init_dataset(config.data)
        self.diffusion = init_diffusion(config.diffusion)
        self.model = init_model(config.model)
        self.loss_object = LossObject(config.loss)

        # init saving utils
        self.logger = Logger(config.logger, config.loss, rank)
        self.model_checkpoints = Checkpoints(config.checkpoints, rank)
        pass

    
    

    def train(
        self,
        process,
        fprint=print
    ):      
        # multiprocessing and device entities
        torch.cuda.set_device(process.gpu)
        self.model.cuda(process.gpu)
        self.diffusion.cuda(process.gpu)

        new_batch_size = 64
        prev_batch_size = self.data_generator.batch_size
        self.data_generator.change_batch_size_in_dataloaders(new_batch_size)
        if   hasattr(self.diffusion, 'precompute_tail_normalization_statistics'):
            self.diffusion.precompute_tail_normalization_statistics(self.data_generator, 2000)
        elif hasattr(self.diffusion, 'precompute_xt_normalization_statistics'):
            self.diffusion.precompute_xt_normalization_statistics(self.data_generator, 2000)
        self.data_generator.change_batch_size_in_dataloaders(prev_batch_size)
        torch.cuda.empty_cache()
        
        if process.distributed:
            self.model = to_distributed(self.model, process.gpu)
            self.data_generator.to_distributed('train', process.rank, process.world_size)
            self.data_generator.change_batch_size_in_dataloaders(
                self.data_generator.batch_size // process.world_size + 1
            )
            fprint(f'process {process.rank} bs: {self.data_generator.dist_bs}')
            train_loader = self.data_generator.distributed_loader
            dist.barrier()
        else:
            train_loader = self.data_generator.train_loader
        
        episode = 0
        self.optimizer = Optimizer(self.model, self.config.optimization_config)
        self.metrics_tracker = MetricsTracker(self.config.metrics_config)

        for epoch in range(self.config.n_epochs):

            # train stage
            
            self.model.train()
            if process.distributed:
                self.data_generator.distributed_sampler.set_epoch(epoch)
            
            for batch_index, x0 in enumerate(train_loader):
                batch = self.data_generator.create_batch(x0, device='cuda')
                trainer_printer(
                    process.rank, episode, 1, epoch, self.config.n_epochs, 
                    batch_index, self.data_generator.num_batches, fprint=fprint
                )
                self.diffusion.train_step(
                    batch=batch,
                    model=self.model,
                    loss_object=self.loss_object,
                    mode='train'
                )
                loss = self.loss_object.compute_loss(
                    batch=batch, 
                    diffusion=self.diffusion, 
                    train_object=None, 
                    mode='train'
                )
                self.optimizer.optimizing_step(
                    loss, 
                    batch_index + epoch * self.data_generator.num_batches
                )
                self.metrics_tracker.compute_metrics(
                    batch, self.diffusion, self,
                    mode='train', episode=episode, epoch=epoch
                )
                del batch

            if process.is_root_process:
                loss = self.loss_object.get_accumulated_loss()
                metrics = self.metrics_tracker.get_accumulated_metrics()
                self.logger.update(loss, metrics, episode, epoch, mode='train')
                self.model_checkpoints.create_checkpoint(
                    self.model, self.optimizer, episode, epoch)
            if process.distributed:
                dist.barrier()

            # validation stage

            if self.metrics_tracker.validation_rule(epoch) and process.is_root_process:
                fprint('')
                self.optimizer.switch_to_ema()
                torch.cuda.empty_cache()
                self.model.eval()
                with torch.no_grad():
                    # go through train and validation datasets with fixed model params
                    for mode, data_loader in zip(
                        [ 'ema_on_train',                   'ema_on_validation'                   ],
                        [ self.data_generator.train_loader, self.data_generator.validation_loader ]
                    ):
                        fprint(f'    <> val stage: mode -> {mode}', flush=True)
                        for batch_index, x0 in enumerate(data_loader):
                            batch = self.data_generator.create_batch(x0, device='cuda')
                            self.diffusion.train_step(
                                batch=batch,
                                model=self.model,
                                loss_object=self.loss_object,
                                mode=mode
                            )
                            loss = self.loss_object.compute_loss(
                                batch=batch, 
                                diffusion=self.diffusion, 
                                train_object=None, 
                                mode=mode
                            )    
                            self.metrics_tracker.compute_metrics(
                                batch, self.diffusion, self,
                                mode=mode, episode=episode, epoch=epoch
                            )
                            del batch

                        loss = self.loss_object.get_accumulated_loss()
                        metrics = self.metrics_tracker.get_accumulated_metrics()
                        self.logger.update(loss, metrics, episode, epoch, mode=mode)

                    self.optimizer.switch_from_ema()

                # plot dynamic
                plot_loss(
                    self.config.logger.folder, lc=0.25, uc=4,
                    graphics_path=self.config.logger.folder
                )
                
                # end validation

            if process.distributed:    
                dist.barrier()
                
            # end of an epoch

        self.optimizer.switch_to_ema()
        self.model.eval()
        self.model.cpu()
        self.diffusion.cpu()

        if process.is_root_process:
            self.logger.save_logs()
            self.model_checkpoints.create_checkpoint(
                self.model, self.optimizer, None, None, last=True)
        
        if process.distributed:
            dist.barrier()
        pass

