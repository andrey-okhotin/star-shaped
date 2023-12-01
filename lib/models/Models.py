from torch.nn.parallel import DistributedDataParallel

from models.ncsnpp_unet.initialization import init_ncsnpp
from models.mlp.initialization import init_mlp
from models.t5.initialization import init_t5encoder



def init_model(config):
    switch = {
        'NCSNpp' : init_ncsnpp,
        'MLP' : init_mlp,
        'T5Encoder' : init_t5encoder
    }
    if config['model_name'] in switch.keys():
        model = switch[config['model_name']](config['model_config'])
        model.device = 'cpu'
        model.model_name = config['model_name']
        model.model_config = config['model_config']
    else:
        raise NotImplementedError
    return model




def to_distributed(model, gpu, process_group=None):
    """
    """
    # save default model params
    model_name = model.model_name
    model_config = model.model_config
    resolution_flag = False
    if 'resolution' in model.__dict__.keys():
        resolution = model.resolution
        resolution_flag = True
    
    model = DistributedDataParallel(
        model, 
        device_ids=[gpu], 
        process_group=process_group
    )
    
    # load default model params to DDP
    model.model_name = model_name
    model.model_config = model_config
    if resolution_flag:
        model.resolution = resolution

    return model
    
    