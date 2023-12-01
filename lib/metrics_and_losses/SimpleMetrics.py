import os
import torch
import numpy as np
from matplotlib import pyplot as plt

from npeet import entropy_estimators as ee




def init_simple_metric(metric_config):
    switch = {
        'loss' : Loss_metric,
        'npeet_kl' : NpeetKL_metric
    }
    metric = switch[metric_config['metric']](metric_config)
    return metric




class SimpleMetric:
    
    def __init__(self, config):
        self.iters, self.metric = [], []
        self.name = config.metric
        self.freq = config.freq
        self.config = config
        
    def __call__(self, train_object, iteration):
        if (iteration+1) % self.freq == 0:
            self.estimation(train_object)
            self.iters.append(iteration+1)
        pass
    
    def estimation(self, train_object):
        raise NotImplementedError
    
    def plot(self, train_object, graphics_path=None):
        raise NotImplementedError
        
    def save(self, train_object, path):
        pack = self.get_data_pack(train_object)
        self.pack_size = len(self.pack)
        for i, t in enumerate(pack):
            torch.save(t, f'{path}_{self.name}_t{i}.pt')
        pass
    
    def load(self, path):
        pack = tuple()
        for i in range(self.pack_size):
            pack += (torch.load(f'{path}_{self.name}_t{i}.pt'),)
        return pack
    
    
    
    
class Loss_metric(SimpleMetric):
    
    def estimation(self, train_object):
        pass
        
    def plot(self, train_object, graphics_path=None):
        if len(train_object.train_loss_list) == 0:
            return None
        lw = 2
        
        fig = plt.figure(figsize=(15,4))
        ax = fig.add_subplot(121)
        plt.grid()
        if min(train_object.train_loss_list) > 0:
            plt.yscale('log')
        
        avg = 100
        eq = min(train_object.train_loss_list)
        if len(train_object.train_loss_list) > avg:
            cur_num_iters = (len(train_object.train_loss_list) // avg) * avg
            cur_iters = torch.arange(cur_num_iters).to(torch.float32)
            loss_dynamic = torch.tensor(train_object.train_loss_list)[:cur_num_iters]
            plt.plot( 
                cur_iters.reshape(-1,avg).mean(dim=1),
                loss_dynamic.reshape(-1,avg).mean(dim=1), 
                color='cornflowerblue', 
                label='NN loss', 
                lw=lw
            )
            eq = loss_dynamic.reshape(-1,avg).mean(dim=1).min().item()
        if len(train_object.ema_val_iters) > 0:
            plt.plot(
                train_object.ema_val_iters, 
                train_object.ema_val_loss_list, 
                color='blue',
                label='EMA loss', 
                lw=lw
            )
            eq = min(train_object.ema_val_loss_list)
        if eq > 0:
            plt.ylim(eq / 10, eq * 20)
        else:
            plt.ylim(eq * 20, eq / 10)
        plt.ylabel('loss', fontsize=16)
        plt.xlabel('iters', fontsize=16)
        plt.legend(loc=1, fontsize=10)
        
        if 'train_mean_loss' in train_object.__dict__.keys():
            ax = fig.add_subplot(122)
            plt.grid()
            plt.yscale('log')
            plt.plot(
                train_object.train_mean_loss[:train_object.diffusion.num_steps], 
                color='cornflowerblue', 
                label='NN loss', 
                lw=lw
            )
            plt.plot(
                train_object.ema_val_mean_loss[:train_object.diffusion.num_steps], 
                color='blue', 
                label='EMA loss', 
                lw=lw
            )
            plt.xlim(-10, train_object.diffusion.num_steps+10)
            plt.xlabel('time', fontsize=16)
            plt.ylabel('loss', fontsize=16)
            plt.legend(loc='best', fontsize=10)
        
        if not (graphics_path is None):
            fig.savefig(
                os.path.join(graphics_path, 'loss.png'), 
                bbox_inches='tight'
            )
        plt.show()
    
    def get_data_pack(self, train_object):
        self.pack = (train_object.train_loss_list,)
        if 'train_mean_loss' in train_object.__dict__.keys():
            self.pack += (
                train_object.train_mean_loss[:train_object.diffusion.num_steps].tolist(),
                train_object.ema_val_mean_loss[:train_object.diffusion.num_steps].tolist()
            )
        if 'ema_val_iters' in train_object.__dict__.keys():
            self.pack += (
                train_object.ema_val_iters,
                train_object.ema_val_loss_list
            )
        return self.pack




def npeet_symmetric_kl_estimation(q_samples, p_samples):
    return (
        (ee.kldiv(q_samples, p_samples, k=10, base=np.e) +
        ee.kldiv(p_samples, q_samples, k=10, base=np.e)) / 2
    )
    
    
    
    
class NpeetKL_metric(SimpleMetric):
    
    def estimation(self, train_object):
        train_object.optimizer.switch_to_ema()
        kl_estimations = torch.zeros((self.config.num_estimations,), dtype=torch.float32)
        for i in range(self.config.num_estimations):
            q_samples = train_object.dataset.sample(
                'train', self.config.num_samples, true_sample=True
            )
            p_samples = train_object.sample_func(
                train_object.model,
                train_object.diffusion,
                self.config.num_samples,
                self.config.batch_size
            )
            if train_object.dataset.dataset_type == 'simplex':
                q_samples, p_samples = q_samples[:,:,:,:-1], p_samples[:,:,:,:-1]
            assert p_samples.isnan().sum().item() / p_samples.flatten().shape[0] < 0.2
            p_samples = torch.nan_to_num(p_samples)
            q_samples = q_samples.reshape(self.config.num_samples, -1)
            p_samples = p_samples.reshape(self.config.num_samples, -1)
            kl_estimations[i] = npeet_symmetric_kl_estimation(q_samples, p_samples)
        train_object.optimizer.switch_from_ema()
        self.metric.append(kl_estimations.mean().item())
        pass
    
    def plot(self, train_object, graphics_path=None):
        if len(self.metric) == 0:
            return None
        best_iter = torch.tensor(self.metric).argmin()
        fig = plt.figure(figsize=(7,4))
        ax = fig.add_subplot(111)
        plt.grid()
        plt.plot(
            self.iters, self.metric, 
            lw=2., 
            color='darkmagenta', 
            label=(
                f"best iter: {self.iters[best_iter]:7d}\n"+
                f"best metric: {self.metric[best_iter]:8.3f}"
            )
        )
        plt.ylim(0, max(
            10 * min(self.metric), 
            1.5 * self.metric[-1])
        )
        plt.legend(loc='best', fontsize=14)
        plt.ylabel('npeet kl', fontsize=16)
        plt.xlabel('iters', fontsize=16)
        if not (graphics_path is None):
            fig.savefig(
                os.path.join(graphics_path, 'npeet_symmetric_kl.png'), 
                bbox_inches='tight'
            )
        plt.show()  
        
    def get_data_pack(self, train_object):
        self.pack = (self.iters, self.metric)
        return self.pack
        
