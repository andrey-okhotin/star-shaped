import os
import ml_collections
import torch

from saving_utils.get_repo_root import get_repo_root
from models.t5.t5encoder import T5EncoderDiffusionModel



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
    if 'default' in model_config.keys():
        return T5EncoderDiffusionModel(base_text8())

    elif 'pretrained_model' in model_config.keys():
        model = T5EncoderDiffusionModel(base_text8())
        if os.path.isabs(model_config['pretrained_model']):
            model_config['pretrained_model'] = os.path.basename(os.path.normpath(model_config['pretrained_model']))
        path = os.path.join(get_repo_root(), '..', 'app', 'pretrained_models', model_config['pretrained_model'])
        if not os.path.exists(path):
            raise ValueError("You need to put model checkpoint in folder \"pretrained_model\" in your current directory.")
        state_dict = torch.load(path, map_location='cpu')
        state_dict = { k.removeprefix('module.') : v for k, v in state_dict.items() }
        model.load_state_dict(state_dict)
        return model
    
    raise NotImplementedError
