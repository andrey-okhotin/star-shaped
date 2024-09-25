from pipelines_configs.TrainingCifar10 import training_cifar10
from pipelines_configs.SamplingCifar10 import sampling_cifar10
from pipelines_configs.TrainingText8 import training_text8
from pipelines_configs.SamplingText8 import sampling_text8
from pipelines_configs.EstimatingNLLText8 import estimating_nll_text8




def init_pipeline(pipeline_name):
    switch = {
        'training_cifar10'               : training_cifar10,
        'sampling_cifar10'               : sampling_cifar10,
        'training_text8'                 : training_text8,
        'sampling_text8'                 : sampling_text8,
        'estimating_nll_text8'           : estimating_nll_text8
    }
    if pipeline_name in switch:
        return switch[pipeline_name]
    raise NotImplementedError




def set_pipelines_arguments(parser):
    """
    DESCRIPTION:
        Add following arguments for all available pipelines. There are three types
        of pipelines: 
            - training
            - sampling
            - negative log-likelihood estimation

        For particular pipeline you need a subset of the following arguments. In other 
        files in this directory you can find predefined pipelines with examples of values 
        of all necessary arguments.
        
        Experiment parameters:
        
            <>  pipeline -> str
                    Name of particular predefined pipeline. Used in all pipelines.
                    
            <>  diffusion -> str
                    Name of diffusion model. Used in all pipelines. Diffusion model
                    define train, sampling and nll estimation procedures. There are
                    predefined configs for all diffusion models reported in the article
                    in the file:
                        - lib/diffusion/default_diffusion_configs.py
            
            <>  pretrained_model -> str
                    Name of file.pt - model.state_dict() of torch.nn.Module. Must
                    be in the directory:
                        - lib/pretrained_models
                    Prefix of model name select predefined NN architecture (so not all 
                    files name are possible). Used in sampling and ll estimation 
                    pipelines.
            
            <>  loss -> str
                    Name of loss function for training pipelines. Loss function can
                    depend on selected diffusion.
            
            <>  num_samples -> int
                    How many objects sample in sampling pipeline.
                    For how many objects estimate nll in nll estimation pipeline. If 
                    you set num_samples in nll pipeline greater than number of objects
                    in selected part of the dataset, than nll will be estimated on all
                    objects in this part of the dataset.
            
            <>  num_sampling_steps -> int
                    How many steps do in generation process in sampling pipeline.
            
            <>  batch_size -> int
                    Batch_size for all pipelines.
                    
            <>  num_iwae_trajectories -> int
                    Num trajectoris for IWAE nll estimation in nll pipeline.
            
            <>  dataset_part -> str
                    On what part of the dataset (train, val or test) estimate nll.
            
        Saving parameters:
        
            <>  logs_file -> str
                    Name of file.txt where processes will write their progress info.
                    This file will be created in the root of the repository.
            
            <>  save_folder -> str
                    General name for folder or file for saving necessary things.
                        - training pipeline: 
                        1) Name of folder where will be saved model checkpoints. This folder 
                        will be created in directory:
                            - checkpoints
                        2) Name of folder where will be saved loss dynamic results. This folder
                        will be created in directory:
                            - results
                        
                        - sampling pipeline: Name of folder where will be saved generated 
                        objects. This folder will be created in directory:
                            - results
                            
                        - nll estimation pipeline: Prefix for files where will be saved nlls 
                        for every object and average nll in file.pt format. These files will
                        be created in directory:
                            - results/nll_estimations
    
    INPUT:
        <>  parser -> argparse.ArgumentParser()

    """
    
    # base entities
    parser.add_argument('-pipeline', '--pipeline', default='', type=str)
    parser.add_argument('-diffusion', '--diffusion', default='', type=str)
    parser.add_argument('-pretrained_model', '--pretrained_model', default='', type=str)
    parser.add_argument('-data', '--data', default='', type=str)
    parser.add_argument('-loss', '--loss', default='', type=str)
    parser.add_argument('-sampling_method', '--sampling_method', default='', type=str)
    parser.add_argument('-num_samples', '--num_samples', default=20, type=int)
    parser.add_argument('-num_sampling_steps', '--num_sampling_steps', default=-1, type=int)
    parser.add_argument('-batch_size', '--batch_size', default=20, type=int)
    parser.add_argument('-num_iwae_trajectories', '--num_iwae_trajectories', type=int)
    parser.add_argument('-dataset_part', '--dataset_part', type=str)

    # saving args
    parser.add_argument('-save_folder', '--save_folder', default='', type=str)
    pass
