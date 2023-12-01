import torch
import numpy as np
from matplotlib import pyplot as plt
from ml_collections import ConfigDict

import os
from pathlib import Path

from IPython.display import clear_output
from IPython.display import IFrame

from saving_utils.Logger import Logger
from saving_utils.get_repo_root import get_repo_root




def plot_img(img, title=''):
    fig = plt.figure(figsize=(6,6))
    fig.patch.set_facecolor('white')
    ax = fig.add_subplot(111)
    plt.axis('off')
    plt.title(title, fontsize=22)
    plt.imshow(img)
    plt.show()
    pass




def get_logs(folder):
    logger_config = ConfigDict({
        'folder' : folder,
        'saving_freq' : None
    })
    loss_config = None
    logger = Logger(logger_config, loss_config, rank=0)
    return logger.load_logs()




def plot_loss(folder, lc=0.9, uc=1.2, graphics_path=None):
    logs = get_logs(folder)
    if not (graphics_path is None):
        folder = os.path.join(get_repo_root(), 'results', graphics_path)
        if not os.path.exists(folder):
            os.mkdir(folder)
    
    lw = 2
    mk = 3.5
    scalar_plot_info = {
        'train' : { 
            'linewidth' : lw / 2, 'color' : 'cornflowerblue', 'label' : 'train_loss' 
        },
        'ema_on_train' : { 
            'linewidth' : lw, 'color' : 'blue', 'label' : 'EMA train loss', 
            'marker' : 'D', 'markersize' : mk
        },
        'ema_on_validation' : { 
            'linewidth' : lw, 'color' : 'red', 'label' : 'EMA validation loss', 
            'marker' : 'D', 'markersize' : mk
        }
    }
    already_plot = set()
    for stage1, stage_metrics1 in logs.items():
        for metric, metric_info in stage_metrics1.items():
            _, epochs, metric_vals = metric_info
            if (
                f'{stage1}:{metric}' in already_plot or 
                len(metric_vals) == 0 or
                not isinstance(metric_vals[0], float)
            ):
                continue
            fig = plt.figure(figsize=(9,4))
            plt.grid()
            plt.title(metric, fontsize=18)
            plt.xlabel('epoch', fontsize=16)
            plt.ylabel('value', fontsize=16)
            for stage2, stage_metrics2 in logs.items():
                if metric in stage_metrics2.keys():
                    _, ep, val = stage_metrics2[metric]
                    if len(val) > 0:
                        plt.ylim(min(val) * lc, min(val) * uc)
                        plt.plot(ep, val, **scalar_plot_info[stage2])
                        already_plot.add(f'{stage2}:{metric}')
            plt.legend(loc=1, fontsize=14)

            if not (graphics_path is None):
                fig.savefig(os.path.join(folder, f'{metric}.png'), bbox_inches='tight')
            plt.show()
    pass
    
    
    
    
def plot_td_metric(
    dynamic,
    ylabel,
    xlabel,
    ylim=None,
    yscale=None,
    graphics_path=None
):
    avg_window = 10
    val_stages = 512 // 8
    epochs_dynamic = dynamic[1][-1:]
    weights_dynamic = dynamic[2][-1:]
    dlen = len(epochs_dynamic)
    step = epochs_dynamic[-1]+1 / dlen
    
    fig = plt.figure(figsize=(8,4))
    ax = fig.add_subplot(111)
    plt.ylabel(ylabel, fontsize=14)
    plt.xlabel(xlabel, fontsize=14)
    for i, epoch, weights in zip(np.arange(dlen), epochs_dynamic, weights_dynamic):
        if epoch == 0:
            label_kwargs = { 'label' : "start training" }
        elif epoch == dlen - 1:
            label_kwargs = { 'label' : "finish training" }
        else:
            label_kwargs = {}
        t, w = weights
        plt.plot(
            torch.tensor(t)[:100], torch.tensor(w)[:100],
            color=(1 - (i+1)/dlen, 0, (i+1)/dlen),
            alpha=0.5,
            lw=0.8,
            **label_kwargs
        )
    ax.set_xticks(torch.arange(0,101,20))
    ax.set_xticklabels([f"t = {i}" for i in range(0,101,20)])
    if not (ylim is None):
        plt.ylim(ylim)
    if not (yscale is None):
        plt.yscale(yscale)
    plt.grid()
    plt.legend(loc='best', fontsize=14)
    
    if not (graphics_path is None):
        fig.savefig(graphics_path+'.png', bbox_inches='tight')
    plt.show()
    return weights_dynamic    
    
    
    

def plot_metrics(folder, graphics_path=None):
    logs = get_logs(folder)
    if not (graphics_path is None):
        folder = os.path.join(get_repo_root(), 'results', graphics_path)
        if not os.path.exists(folder):
            os.mkdir(folder)
        graphics_path = os.path.join(folder, 'KL_for_timepoints')
    else:
        graphics_path = None
    plot_td_metric(
        dynamic=logs['train']['KL_rescaled'],
        ylabel='KL_rescaled terms',
        xlabel='time',
        graphics_path=graphics_path+'_rescaled'
    )
    plot_td_metric(
        dynamic=logs['train']['KL'],
        ylabel='KL terms',
        xlabel='time',
        graphics_path=graphics_path
    )
    pass



