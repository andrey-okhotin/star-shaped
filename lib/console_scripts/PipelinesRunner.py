import os
import sys
import argparse
import traceback
import time
from ml_collections import ConfigDict

import torch
import torch.multiprocessing as mp
import torch.distributed as dist

# run only from 'star-shaped' folder
sys.path.append(os.path.join(os.getcwd(), 'lib'))

from console_scripts.cluster_utils import logger_print, logger_reset
from pipelines.fix_seed import fix_seed
from pipelines_configs.init_pipeline import init_pipeline, set_pipelines_arguments




def pipelines_runner(rank, args):
    """
    DESCRIPTION:
        Define necessary attributes of the current process. Init
        log file where all pipelines processes write info about their
        progress and errors. After that fix seed for current process and 
        run pipeline.
        
    """
    # multiprocessing part
    process = ConfigDict()
    process.rank = args.nr
    process.world_size = len(args.gpu)
    process.is_root_process = (process.rank == 0)
    process.gpu = args.gpu[process.rank]
    process.distributed = (len(args.gpu) > 1)
    if process.distributed:
        dist.init_process_group(                                   
            backend='nccl',                                   
            world_size=process.world_size,                              
            rank=process.rank                                               
        )

    # init logger for tracking pipeline progress
    logger_print.file = args.logs_file
    if process.is_root_process:
        logger_reset()
    else:
        time.sleep(3)
    logger_print(f'proc: {process.rank} gpu: {process.gpu}')

    # run pipeline
    logger_print(f'proc: {process.rank} start pipeline')
    pipeline = init_pipeline(args.pipeline)
    try:
        fix_seed(12345 + 100 * process.rank)
        pipeline(process, args)
    except Exception as e:
        logger_print(traceback.format_exc())
        dist.destroy_process_group()
    logger_print(f'proc: {process.rank} finish pipeline')
    pass




def main():
    """
    DESCRIPTION:
        Set multiprocessing arguments for pytorch DistributedDataParallel:
            <>  gpu -> str of format <idx0>_<idx1>_..._<idxk> (example: 0_1_2).
                    Set indices of gpus for pipeline. Code in run_pipeline.py has 
                    already spawned processes for each gpu.
            <>  port -> str
                    Set port for syncronization. Usually work with default arguments, but if not,
                    you need to find available port on your system.
        
        After that set experiment arguments, port for sync and call mp.spawn.
    
    """
    torch.multiprocessing.freeze_support()
    parser = argparse.ArgumentParser()
    device = os.environ.get("args.gpu", "")
    
    if device == "":
        parser.add_argument('-gpu', '--gpu', default='', type=str)
        parser.add_argument('-port', '--port', default='8900', type=str)
        set_pipelines_arguments(parser, args_from='cmd')
    else:
        args = ConfigDict({})
        args.gpu = device
        args.port = os.environ.get('args.port', '8900')
        set_pipelines_arguments(args, args_from='environ')

    parser.add_argument('-nr', '--nr', default=0, type=int)
    cmd_args = parser.parse_args()
    if device == "":
        args = cmd_args
        print(f"\n\n      cmd : {args.gpu}   \n\n", flush=True)
    else:
        args.nr = cmd_args.nr
        print(f"\n\n      environ : {args.gpu}    \n\n", flush=True)
    args.gpu = tuple(map(int, args.gpu.split('_')))   

    # spawn processes with defined pipeline
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = args.port
    mp.set_start_method('fork')
    mp.spawn(pipelines_runner, nprocs=1, args=(args,))
    pass




if __name__ == '__main__':    
    main()



