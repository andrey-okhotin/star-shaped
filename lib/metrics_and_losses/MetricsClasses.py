import torch


    
    
class TimeDependenceMetric:
    
    def __init__(self):
        self.metric_val = torch.zeros((1000,20), dtype=torch.float32)
        self.is_loss = False
    
    def batch_metric(self, batch, diffusion):
        raise NotImplementedError
    
    def batch_metric_update(self, batch, diffusion, train_object, mode):
        batch_t, batch_metric = self.batch_metric(batch, diffusion, train_object, mode)
        self.last_batch_metric = batch_metric.clone().detach()
        self.metric_val[batch_t,1:] = self.metric_val[batch_t,:-1]
        self.metric_val[batch_t,0] = batch_metric.clone().detach().cpu()
        if (
            self.is_loss and 
            'loss_weights' in diffusion.time_distribution.__dict__.keys() and 
            mode == 'train'
        ):
            batch_metric = diffusion.time_distribution.loss_weights[batch_t] * batch_metric
        return batch_metric.mean()
    
    def get_metric(self):
        metric_val = self.metric_val.mean(dim=1)
        timesteps = torch.arange(self.metric_val.shape[0], dtype=torch.float32)
        is_loss = self.is_loss
        self.__init__()
        self.is_loss = is_loss
        if self.is_loss:
            return metric_val.mean()
        return torch.stack((timesteps, metric_val))
    
    


class KL(TimeDependenceMetric):
    def batch_metric(self, batch, diffusion, train_object, mode):
        return (
            batch['t'],
            diffusion.kl(batch).reshape(batch['batch_size'], -1).mean(dim=1)
        )
    

    
    
class KL_rescaled(TimeDependenceMetric):
    def batch_metric(self, batch, diffusion, train_object, mode):
        return (
            batch['t'],
            diffusion.kl_rescaled(batch).reshape(batch['batch_size'], -1).mean(dim=1)
        )




class NLL:
    
    def __init__(self):
        self.metric_val = 0.
        self.num_batches = 0

    def batch_metric(self, batch, diffusion, train_object, mode):
        return (-1) * batch['log_probs'].mean()
    
    def batch_metric_update(self, batch, diffusion, train_object, mode):
        batch_metric = self.batch_metric(batch, diffusion, train_object, mode)
        self.metric_val += batch_metric.detach().cpu()
        self.num_batches += 1
        return batch_metric
    
    def get_metric(self):
        if self.num_batches == 0:
            return None
        metric_val = self.metric_val / self.num_batches
        self.__init__()
        return metric_val
    
