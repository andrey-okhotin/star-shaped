import os
import ml_collections
import torch

from saving_utils.get_repo_root import get_repo_root
from models.ncsnpp_unet.models.ncsnpp import NCSNpp




def default_cifar10_config():
    cifar10_config = ml_collections.ConfigDict()
    cifar10_config.dropout = 0.1
    cifar10_config.embedding_type = 'fourier'
    cifar10_config.name = 'ncsnpp'
    cifar10_config.scale_by_sigma = False
    cifar10_config.ema_rate = 0.9999
    cifar10_config.normalization = 'GroupNorm'
    cifar10_config.nonlinearity = 'swish'
    cifar10_config.nf = 128
    cifar10_config.ch_mult = (1, 2, 2, 2)
    cifar10_config.num_res_blocks = 4
    cifar10_config.attn_resolutions = (16,)
    cifar10_config.resamp_with_conv = True
    cifar10_config.conditional = True
    cifar10_config.fir = False
    cifar10_config.fir_kernel = [1, 3, 3, 1]
    cifar10_config.skip_rescale = True
    cifar10_config.resblock_type = 'biggan'
    cifar10_config.progressive = 'none'
    cifar10_config.progressive_input = 'none'
    cifar10_config.progressive_combine = 'sum'
    cifar10_config.attention_type = 'ddpm'
    cifar10_config.init_scale = 0.0
    cifar10_config.embedding_type = 'positional'
    cifar10_config.fourier_scale = 16
    cifar10_config.conv_size = 3
    cifar10_config.resolution = 32
    return cifar10_config




def init_ncsnpp(model_config):
    switch_default_configs = {
        'ncsnpp-cifar10' : default_cifar10_config
    }

    if 'default' in model_config.keys():
        cifar10_config = switch_default_configs[model_config['default']]()
        return NCSNpp(cifar10_config)

    elif 'pretrained_model' in model_config.keys():
        model_domain = model_config['pretrained_model'].split('_')[0]
        model = NCSNpp(switch_default_configs[model_domain]())
        path = os.path.join(get_repo_root(), 'pretrained_models', model_config['pretrained_model'])
        state_dict = torch.load(path, map_location='cpu')
        state_dict = { k.removeprefix('module.') : v for k, v in state_dict.items() }
        model.load_state_dict(state_dict)
        return model
    
    raise NotImplementedError
