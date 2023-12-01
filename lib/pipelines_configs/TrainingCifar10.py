from ml_collections import ConfigDict

from console_scripts.cluster_utils import logger_print, save_config
from pipelines.UniversalDiffusionTrainer import UniversalDiffusionTrainer
from diffusion.default_diffusion_configs import get_default_diffusion




def training_cifar10(process, args):
    """
    DESCRIPTION:
        Pipeline used for training selected diffusion model on CIFAR10 dataset 
        with NCSN++ architecture. This pipeline used for results reported in the article.
    
    EXAMPLE 1:
    
    >> python lib/run_pipeline.py  
                                **system args** 
        -gpu 0_1_2_3
        -port 8900
        -pipeline training_cifar10
                                **pipeline args**
        -diffusion beta_ss
        -loss KL_rescaled
        -save_folder training_beta_ss_cifar10
        -logs_file logs_training_beta_ss.txt


    EXAMPLE 2 (if you have > 40Gb on single gpu):
    
    >> python lib/run_pipeline.py  
                                **system args** 
        -gpu 0
        -port 8900
        -pipeline training_cifar10
                                **pipeline args**
        -diffusion ddpm
        -loss KL_rescaled
        -save_folder training_ddpm_cosine_cifar10
        -logs_file logs_training_ddpm.txt
        
    """
    # fixed part of pipeline config
    config = ConfigDict({})
    config.model = ConfigDict({
        'model_name' : 'NCSNpp',
        'model_config' : ConfigDict({
            'default' : 'ncsnpp-cifar10'
        })
    })
    config.data = ConfigDict({
        'name'        : 'cifar10',
        'batch_size'  : 128,
        'num_workers' : 4
    })
    config.n_epochs = 2001
    config.optimization_config = ConfigDict({
        'optimizer' : {
            'method' : 'Adam',
            'config' : { 'lr' : 2e-4 }
        },
        'schedulers' : [
            (
                'Linear',
                { 
                    'start_factor' : 0.0002,
                    'end_factor' : 1.0,
                    'total_iters' : 5000
                },
                (0, 5000)
            )
        ],
        'ema' : {
            'ema_rate' : 0.9999,
            'initial_acceleration' : True
        },
        'clip_grad_norm' : 1.0
    })
    config.metrics_config = ConfigDict({
        'validation_freq' : 100,
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
        'saving_freq' : 10,
        'reset_previous' : True
    })
    config.logs_file = args.logs_file
    
    save_config(config, process.rank)
    trainer = UniversalDiffusionTrainer(
        config, 
        process.rank
    )
    trainer.train(
        process,
        fprint=logger_print
    )
    pass



