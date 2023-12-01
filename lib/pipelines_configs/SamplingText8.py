from ml_collections import ConfigDict

from pipelines.UniversalDiffusionSampler import UniversalDiffusionSampler
from console_scripts.cluster_utils import logger_print, save_config, memory_reservation
from diffusion.default_diffusion_configs import get_default_diffusion




def choose_max_batch_size(total_memory):
    return round(total_memory * 300 / 10.5)




def sampling_text8(process, args):
    """
    DESCRIPTION:
        Pipeline used for sampling objects from selected diffusion model trained 
        on Text8 dataset with Base T5-encoder architecture. This pipeline used 
        for results reported in the article.
    
    EXAMPLE 1:
    
    >> python lib/run_pipeline.py  
                                **system args**
        -gpu 0_2_3
        -port 8900
        -pipeline sampling_text8
                                **pipeline args**
        -diffusion categorical_ss
        -pretrained_model t5base-text8_categorical-ss_fully-trained.pt
        -num_samples 100
        -save_folder sampling_categorical_ss_text8
        -logs_file logs_sampling_categorical_ss.txt
        
    """
    # memory reservation and choosing optimal batch size
    max_memory_usage = memory_reservation(process.gpu, logger_print)
    batch_size = choose_max_batch_size(max_memory_usage)
    num_samples_per_node = args.num_samples // process.world_size + 1
    batch_size = min(batch_size, num_samples_per_node)
    logger_print(f'proc: {process.rank} batch_size: {batch_size}')

    # pipeline config
    config = ConfigDict({})
    config.model = ConfigDict({
        'model_name' : 'T5Encoder',
        'model_config' : ConfigDict({
            'pretrained_model' : args.pretrained_model
        })
    })
    config.data = ConfigDict({
        'name'        : 'text8',
        'seq_len'     : 256,
        'batch_size'  : batch_size,
        'num_workers' : 4
    })
    config.diffusion = get_default_diffusion(args.diffusion)
    config.num_sampling_steps = args.num_sampling_steps
    config.save_folder = args.save_folder
    config.logs_file = args.logs_file
    
    save_config(config, process.rank)
    sampler = UniversalDiffusionSampler(config, process)
    sampler.sample(
        num_samples=num_samples_per_node,
        process=process,
        fprint=logger_print
    )
    pass


