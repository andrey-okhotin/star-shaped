import torch
import torch.nn as nn
from torch.nn import functional as F

from metrics_and_losses.MetricsClasses import (
    KL,
    KL_rescaled,
    NLL
)



class LossObject:

    def __init__(self, loss_config):
        """
        INPUT:
        <>  loss_config = {
                'method' : (str) - loss function name
            }
        """
        switch_loss = {
            'KL' : KL,
            'KL_rescaled' : KL_rescaled,
            'NLL' : NLL
        }
        self.loss_obj = switch_loss[loss_config.method]()
        self.loss_obj.is_loss = True
        pass


    def compute_loss(self, batch, diffusion, train_object, mode):
        return self.loss_obj.batch_metric_update(batch, diffusion, train_object, mode)


    def get_accumulated_loss(self):
        return self.loss_obj.get_metric()

