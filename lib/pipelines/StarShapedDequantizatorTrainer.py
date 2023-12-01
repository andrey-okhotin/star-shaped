import torch
from ml_collections import ConfigDict

from visualization_utils.ProgressPrinters import trainer_printer

from data.Datasets import init_dataset
from diffusion.Diffusion import init_diffusion
from models.Models import init_model
from metrics_and_losses.Loss import LossObject
from optimizers.Optimizers import Optimizer
from metrics_and_losses.Metrics import MetricsTracker
from saving_utils.Logger import Logger
from saving_utils.Checkpoints import Checkpoints




class StarShapedDequantizatorTrainer:

    def __init__(self, config, rank=0):
        self.config = config
        self.data_generator = init_dataset(config.data)
        self.model = init_model(config.model)
        self.loss_object = LossObject(ConfigDict({
            'method' : 'NLL'
        }))
        
        self.diffusion = init_diffusion(config.diffusion)
        config.dequantizator.model_config.diffusion = self.diffusion
        self.dequantizator = init_model(config.dequantizator)
        self.diffusion.set_dequantizator(self.dequantizator)

        # init saving utils
        self.logger = Logger(config.logger, config.loss, rank)
        self.dequantizator_checkpoints = Checkpoints(config.checkpoints, rank)
        pass

    
    

    def train(
        self,
        process,
        fprint=print
    ):
        # multiprocessing and device entities
        torch.cuda.set_device(process.gpu)
        self.model.cuda(process.gpu)
        self.model.eval()
        self.dequantizator.cuda(process.gpu)
        self.diffusion.cuda(process.gpu)
 
        new_batch_size = 64
        prev_batch_size = self.data_generator.batch_size
        self.data_generator.change_batch_size_in_dataloaders(new_batch_size)
        if   hasattr(self.diffusion, 'precompute_tail_normalization_statistics'):
            self.diffusion.precompute_tail_normalization_statistics(self.data_generator, 2000)
        elif hasattr(self.diffusion, 'precompute_xt_normalization_statistics'):
            self.diffusion.precompute_xt_normalization_statistics(self.data_generator, 2000)
        self.data_generator.change_batch_size_in_dataloaders(prev_batch_size)

        episode = 0
        self.optimizer = Optimizer(self.model, self.config.optimization_config)
        self.metrics_tracker = MetricsTracker(self.config.metrics_config)

        for epoch in range(self.config.n_epochs):
            for batch_index, x0 in enumerate(self.data_generator.train_loader):
                batch = self.data_generator.create_batch(x0, device='cuda')
                trainer_printer(
                    process.rank, episode, 1, epoch, self.config.n_epochs, 
                    batch_index, self.data_generator.num_batches, fprint=fprint
                )
                # set t = 1 and sample G_1
                batch['t'] = self.diffusion.time_distribution.get_time_points_tensor(
                    batch, t=0
                )
                batch['Gt'] = self.diffusion.sample_Gt(
                    batch['x0'],
                    batch['t']
                )
                with torch.no_grad():
                    batch['x0_prediction'] = self.diffusion.model_prediction(
                        self.model,
                        batch['Gt'],
                        batch['t']
                    )
                batch['log_probs'] = self.diffusion.dequantization(
                    batch['Gt'],
                    batch['x0_prediction'],
                    batch['x0'],
                    batch['t']
                )
                loss = self.loss_object.compute_loss(
                    batch=batch, 
                    diffusion=self.diffusion, 
                    train_object=None,
                    mode='train'
                )
                self.optimizer.optimizing_step(
                    loss, batch_index + epoch * self.data_generator.num_batches)
                self.metrics_tracker.compute_metrics(
                    batch, self.diffusion, self,
                    mode='train', episode=episode, epoch=epoch
                )
                del batch

            loss = self.loss_object.get_accumulated_loss()
            metrics = self.metrics_tracker.get_accumulated_metrics()
            self.logger.update(loss, metrics, episode, epoch, mode='train')
            self.dequantizator_checkpoints.create_checkpoint(
                self.dequantizator, self.optimizer, episode, epoch)
            # end of an epoch

        self.optimizer.switch_to_ema()
        fprint('\n', flush=True)
        self.model.cpu()
        self.dequantizator.cpu()
        self.diffusion.cpu()
        self.logger.save_logs()
        self.dequantizator_checkpoints.create_checkpoint(
            self.dequantizator, self.optimizer, None, None, last=True)
        pass

