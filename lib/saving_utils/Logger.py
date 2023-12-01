import torch
# from torch.utils.tensorboard import SummaryWriter

import json
import os
import shutil
from copy import deepcopy
import numpy as np
from collections import defaultdict

from saving_utils.get_repo_root import get_repo_root




class Logger:
    
    def __init__(self, logger_config, loss_config, rank):
        """
        INPUT:
            
        <>  logger_config = {
                'saving_freq' : (int) - logs will be saved every 'saving_freq' epochs
                'logs_folder' : (str) - path to folder where logs will be saved
            }
        """
        self.rank = rank
        if self.rank != 0:
            return None
        
        # main part
        if not (loss_config is None):
            self.loss_name = 'loss::'+loss_config['method']
            self.experiment_process = { 
                'train'             : { self.loss_name : [ [], [], [] ] },
                'ema_on_train'      : { self.loss_name : [ [], [], [] ] },
                'ema_on_validation' : { self.loss_name : [ [], [], [] ] },
            }
        freq = logger_config['saving_freq']
        self.saving_rule = lambda epoch: isinstance(epoch, int) and (epoch % freq == 0)
        
        logs_path = os.path.join(get_repo_root(), 'logs')
        self.logs_folder = os.path.join(logs_path, logger_config['folder'])
        if ('reset_previous' in logger_config.keys() and 
            logger_config['reset_previous'] and os.path.exists(self.logs_folder)):
            shutil.rmtree(self.logs_folder)
        if not os.path.exists(self.logs_folder):
            os.mkdir(self.logs_folder)
        pass
        
    
    def update(self, loss, metrics, episode, epoch, mode, last=False):
        if self.rank != 0:
            return None
        self.experiment_process[mode][self.loss_name][0].append(episode)
        self.experiment_process[mode][self.loss_name][1].append(epoch)
        self.experiment_process[mode][self.loss_name][2].append(loss.item())
        for metric_name, metric_val in metrics.items():
            if metric_val is not None:
                if not (metric_name in self.experiment_process[mode].keys()):
                    self.experiment_process[mode][metric_name] = [ [], [], [] ]
                self.experiment_process[mode][metric_name][0].append(episode)
                self.experiment_process[mode][metric_name][1].append(epoch)
                if isinstance(metric_val, torch.Tensor):
                    metric_val = metric_val.tolist()
                self.experiment_process[mode][metric_name][2].append(metric_val)
        if self.saving_rule(epoch) or last:
            self.save_logs()
        return self.experiment_process
    
    
    def save_logs(self):
        if self.rank != 0:
            return None
        saving_file = os.path.join(self.logs_folder, 'logs.json')
        with open(saving_file, "w") as json_file:
            json.dump(self.experiment_process, json_file)
        return self.experiment_process
    
    
    def load_logs(self):
        if self.rank != 0:
            return None
        loading_file = os.path.join(self.logs_folder, 'logs.json')
        with open(loading_file) as json_file:
            self.experiment_process = json.load(json_file)
        return self.experiment_process
    
    
    def logs2numpy(self):
        numpy_dict = {}
        for column_name, column_data in self.experiment_process.items():
            numpy_dict[column_name] = np.array(column_data)
        return numpy_dict
    
    

        