from diffusion.synthetic_data_diffusion.dirichlet_star_shaped import DirichletStarShaped
from diffusion.synthetic_data_diffusion.wishart_star_shaped import WishartStarShaped
from diffusion.synthetic_data_diffusion.synthetic_ddpm_diffusion import SyntheticDDPM

from diffusion.geodesic_data_diffusion.von_mises_fisher_star_shaped import VonMisesFisherStarShaped

from diffusion.image_data_diffusion.ddpm_diffusion import DDPM
from diffusion.image_data_diffusion.beta_star_shaped import BetaStarShaped

from diffusion.text_data_diffusion.d3pm_diffusion import D3PM
from diffusion.text_data_diffusion.categorical_star_shaped import CategoricalStarShaped




def init_diffusion(diffusion_config):
    """
    DESCRIPTION:
        Init diffusion model with all necessary methods. You can find predefined configs 
        for all experiments from the paper in file:
            - lib/diffusion/default_diffusion_configs.py
    
    INPUT:
        <>  diffusion_config -> ml_collections.ConfigDict = {
                method       -> str:     name of diffusion model
                object_shape -> tuple:   single object shape. For cifar10: (3,32,32) 
                num_steps    -> int:     discretization for diffusion model
                scheduler    -> str:     name of the noise schedule to use
            }
    
    OUTPUT:
        <>  diffusion -> object that defines all operations with diffusion model
    
    """
    switch = {
        'DirichletStarShaped'      : DirichletStarShaped,
        'WishartStarShaped'        : WishartStarShaped,
        'SyntheticDDPM'            : SyntheticDDPM,
        
        'VonMisesFisherStarShaped' : VonMisesFisherStarShaped,
        
        'DDPM'                     : DDPM,
        'BetaStarShaped'           : BetaStarShaped,

        'D3PM'                     : D3PM,
        'CategoricalStarShaped'    : CategoricalStarShaped,
    }
    method = diffusion_config['method']
    
    if method in switch:
        diffusion = switch[method](diffusion_config)
    else:
        raise NotImplementedError
    return diffusion
