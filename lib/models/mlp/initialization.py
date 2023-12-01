import os
import ml_collections
import torch

from saving_utils.get_repo_root import get_repo_root
from models.mlp.mlp import MLP




def default_mlp_config():
    config = ml_collections.ConfigDict()
    config.hidden_dims = [ 512 ] * 3
    config.act = 'swish'
    config.time_embed_dim = 32
    return config




def default_pdm_2x2_mlp_config():
    config = default_mlp_config()
    config.domain = 'pdm_2x2'
    config.input_dim = 4
    config.output_dim = 3
    return config




def default_simplex_mlp_config():
    config = default_mlp_config()
    config.domain = 'simplex'
    config.input_dim = 3
    config.output_dim = 3
    return config




def default_sphere_mlp_config():
    config = default_mlp_config()
    config.domain = 'sphere'
    config.input_dim = 3
    config.output_dim = 3
    return config




def init_mlp(model_config):
    switch_default_configs = {
        'mlp-ellipse' : default_pdm_2x2_mlp_config,
        'mlp-simplex' : default_simplex_mlp_config,
        'mlp-sphere'  : default_sphere_mlp_config
    }
    
    if   'default' in model_config.keys():
        config = switch_default_configs[model_config['default']]()
        return MLP(config)
    
    elif 'pretrained_model' in model_config.keys():
        model_domain = model_config['pretrained_model'].split('_')[0]
        model = MLP(switch_default_configs[model_domain]())
        path = os.path.join(get_repo_root(), 'pretrained_models', model_config['pretrained_model'])
        model.load_state_dict(torch.load(path, map_location='cpu'))
        return model
    
    raise NotImplementedError
