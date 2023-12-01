import torch
from matplotlib import pyplot as plt
import os
from pathlib import Path
from IPython.display import clear_output
from IPython.display import IFrame

from saving_utils.get_repo_root import get_repo_root




def steps_printer(
    index, 
    num_iterations, 
    time_point, 
    num_time_points, 
    mode='jupyter', 
    gpu=0,
    fprint=print
):
    if mode == 'jupyter':
        fprint(f'\r   gpu: {gpu}    iteration: {int(index+1):3d}/{int(num_iterations):3d},', end='')
        fprint(f'    t={int(time_point+1):4d}/{num_time_points:4d}', end='')
    elif mode == 'bash':
        if (time_point+1) % steps_printer.print_freq == 0:
            s = f'\n   gpu: {gpu}   iter: {int(index+1):3d}/{int(num_iterations):3d}'
            s = s + f'   t={int(time_point+1):4d}/{int(num_time_points):4d}'
            fprint(s, end='', flush=True)
    pass




def trainer_printer(
    rank, 
    episode, num_episodes, 
    epoch, num_epochs, 
    batch_index, num_batches,
    fprint=print
):
    if rank == 0:
        output_str = ""
        output_str = output_str + f'    episode: {episode:3d}/{num_episodes:3d},'
        output_str = output_str + f'    epoch: {epoch:4d}/{num_epochs:4d},'
        output_str = output_str + f'    batch: {batch_index:4d}/{num_batches:4d}'
        fprint('\r'+output_str, end='', flush=True)
    pass
  


    
def single_print(obj, flag):
    if single_print.mode == 1:
        folder = os.path.join(get_repo_root(), f'results/logs_{flag}.pt')
        torch.save(obj, folder)
        single_print.mode -= 1
        print('\n____\n|\n|\n|\n|  SUCCESFULL! \n|\n|\n|_____')
        assert 0 == 1
    elif single_print.mode > 1:
        single_print.mode -= 1
    pass
single_print.mode = 1
    

            
    