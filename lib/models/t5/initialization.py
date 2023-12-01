import os
import ml_collections
import torch

from saving_utils.get_repo_root import get_repo_root
from models.t5.t5encoder import T5EncoderDiffusionModel




def small_text8():
    t5_config = ml_collections.ConfigDict()
    t5_config.vocab_size = 27
    t5_config.predict_var = False
    return t5_config




def base_text8():
    t5_config = ml_collections.ConfigDict()
    t5_config.vocab_size = 27
    t5_config.num_layers = 12
    t5_config.d_ff = 3072
    t5_config.num_heads = 12
    t5_config.d_model = 768
    t5_config.predict_var = False
    return t5_config




def init_t5encoder(model_config):
    switch_default_configs = {
        't5small-text8' : small_text8,
        't5base-text8'  : base_text8
    }
    
    if 'default' in model_config.keys():
        config = switch_default_configs[model_config['default']]()
        return T5EncoderDiffusionModel(config)

    elif 'pretrained_model' in model_config.keys():
        model_domain = model_config['pretrained_model'].split('_')[0]
        model = T5EncoderDiffusionModel(switch_default_configs[model_domain]())
        path = os.path.join(get_repo_root(), 'pretrained_models', model_config['pretrained_model'])
        state_dict = torch.load(path, map_location='cpu')
        state_dict = { k.removeprefix('module.') : v for k, v in state_dict.items() }
        model.load_state_dict(state_dict)
        return model
    
    raise NotImplementedError
