from ml_collections import ConfigDict

from console_scripts.cluster_utils import logger_print, save_config
from pipelines.UniversalDiffusionTrainer import UniversalDiffusionTrainer
from diffusion.default_diffusion_configs import get_default_diffusion




def training_text8(process, args):
    """
    DESCRIPTION:
        Pipeline used for training selected diffusion model on Text8 dataset with 
        Base T5-encoder architecture. This pipeline used for results reported in the article.
    
    EXAMPLE 1 (example for training on 3 A100):
    
    >> python lib/run_pipeline.py  
                                **system args** 
        -gpu 0_1_2
        -port 8900
        -pipeline training_text8
                                **pipeline args**
        -diffusion categorical_ss
        -loss KL
        -save_folder training_categorical_ss_text8
        -logs_file logs_training_categorical_ss.txt


    EXAMPLE 2 (if you have > 160Gb on single gpu):
    
    >> python lib/run_pipeline.py  
                                **system args** 
        -gpu 0
        -port 8900
        -pipeline training_text8
                                **pipeline args**
        -diffusion d3pm
        -loss KL
        -save_folder training_d3pm_text8
        -logs_file logs_training_d3pm.txt
    
    """
    # fixed part of pipeline config
    config = ConfigDict({})
    config.model = ConfigDict({
        'model_name' : 'T5Encoder',
        'model_config' : ConfigDict({
            'default' : 't5base-text8'
        })
    })
    config.data = ConfigDict({
        'name'        : 'text8',
        'seq_len'     : 256,
        'batch_size'  : 512,
        'num_workers' : 4
    })
    config.n_epochs = 2048
    config.optimization_config = ConfigDict({
        'optimizer' : {
            'method' : 'Adam',
            'config' : { 'lr' : 5e-4 }
        },
        'schedulers' : [
            (
                'Linear',
                { 
                    'start_factor' : 0.0002,
                    'end_factor'   : 1.0,
                    'total_iters'  : 10000
                },
                (0, 10000)
            ),
            (
                'Exponential',
                {
                    'gamma' : 0.999997
                },
                (10000, 1e+9)
            )
        ],
        'ema' : {
            'ema_rate' : 0.9999,
            'initial_acceleration' : True
        },
        'clip_grad_norm' : 1.0
    })
    config.metrics_config = ConfigDict({
        'validation_freq' : 32,
        'metrics_list' : []
    })

    # variable part of pipeline config
    config.diffusion = get_default_diffusion(args.diffusion)
    config.loss = ConfigDict({
        'method' : args.loss
    })
    config.logger = ConfigDict({
        'folder' : args.save_folder,
        'saving_freq' : 1,
        'reset_previous' : True
    })
    config.checkpoints = ConfigDict({
        'folder' : args.save_folder,
        'saving_freq' : 32,
        'reset_previous' : True
    })
    config.logs_file = args.logs_file
    
    save_config(config, process.rank)
    trainer = UniversalDiffusionTrainer(
        config, 
        rank=process.rank
    )
    trainer.train(
        process,
        fprint=logger_print
    )
    pass



