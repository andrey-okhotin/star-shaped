from ml_collections import ConfigDict

from console_scripts.cluster_utils import logger_print, save_config
from pipelines.UniversalNLLEstimator import UniversalNLLEstimator
from diffusion.default_diffusion_configs import get_default_diffusion




def estimating_nll_text8(process, args):
    """
    DESCRIPTION:
        Pipeline used for estimating NLL on Text8 dataset with sequence length of 256
        with selected diffusion model and Base T5-encoder architecture. This pipeline 
        used for results reported in the article.
    
    EXAMPLE 1:
    
    >> python lib/run_pipeline.py  
                                **system args** 
        -gpu 0_1_2
        -port 8900
                                **pipeline args**
        -pipeline estimating_nll_text8
        -diffusion categorical_ss
        -pretrained_model t5base-text8_categorical-ss_fully-trained.pt
        -num_samples 20
        -batch_size 20
        -dataset_part test
        -num_iwae_trajectories 1
        -save_folder nll_text8_categorical-ss
        -logs_file logs_nll_text8_categorical_ss.txt

    """
    config = ConfigDict({})
    config.data = ConfigDict({
        'name'        : 'text8',
        'seq_len'     : 256,
        'batch_size'  : args.batch_size,
        'num_workers' : 4
    })
    config.model = ConfigDict({
        'model_name' : 'T5Encoder',
        'model_config' : ConfigDict({
            'pretrained_model' : args.pretrained_model
        })
    })
    config.diffusion = get_default_diffusion(args.diffusion)
    config.save_folder = args.save_folder
    config.logs_file = args.logs_file
    if args.num_samples == -1:
        num_samples = int(1e+9)
    else:
        num_samples = args.num_samples
    
    save_config(config, process.rank)
    nll_estimator = UniversalNLLEstimator(config)
    nll_estimator.estimate(
        num_samples,
        args.dataset_part,
        args.num_iwae_trajectories,
        process,
        fprint=logger_print
    )
    pass



