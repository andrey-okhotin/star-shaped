from ml_collections import ConfigDict



# diffusion on synthetic data

dirichlet_ss_config = ConfigDict({
    'method'       : 'DirichletStarShaped',
    'object_shape' : (1, 1, 3),
    'num_steps'    : 64,
    'scheduler'    : 'default_dirichlet'
})

wishart_ss_config = ConfigDict({
    'method'       : 'WishartStarShaped',
    'object_shape' : (1, 2, 2),
    'num_steps'    : 64,
    'scheduler'    : 'default_wishart'
})



# diffusion on geodesic data

vmf_ss_config = ConfigDict({
    'method'       : 'VonMisesFisherStarShaped',
    'object_shape' : (1, 1, 3),
    'num_steps'    : 100,
    'scheduler'    : 'default_von_mises_fisher'
})



# diffusion on texts

d3pm_config = ConfigDict({
    'method'       : 'D3PM',
    'object_shape' : (1, 256, 27),
    'num_steps'    : 1000,
    'scheduler'    : 'default_d3pm'
})

categorical_ss_config = ConfigDict({
    'method'       : 'CategoricalStarShaped',
    'object_shape' : (1, 256, 27),
    'num_steps'    : 1000,
    'scheduler'    : 'categorical_ss_as_d3pm'
})



# diffusion on images

ddpm_config = ConfigDict({
    'method'       : 'DDPM',
    'object_shape' : (3, 32, 32),
    'num_steps'    : 1000,
    'scheduler'    : 'cosine',
    'use_norm'     : False
})

gaussian_ss_config = ConfigDict({
    'method'       : 'GaussianStarShaped',
    'object_shape' : (3, 32, 32),
    'num_steps'    : 1000,
    'scheduler'    : 'gauss_ss_as_cosine_ddpm'
})

beta_ss_config = ConfigDict({
    'method'       : 'BetaStarShaped',
    'object_shape' : (3, 32, 32),
    'num_steps'    : 1000,
    'scheduler'    : 'beta_ss_as_cosine_ddpm'
})




def get_default_diffusion(diffusion_name):
    switch = {
        'dirichlet_ss'   : dirichlet_ss_config,
        'wishart_ss'     : wishart_ss_config,
        'vmf_ss'         : vmf_ss_config,
        'd3pm'           : d3pm_config,
        'categorical_ss' : categorical_ss_config,
        'ddpm'           : ddpm_config,
        'gaussian_ss'    : gaussian_ss_config,
        'beta_ss'        : beta_ss_config
    }
    if diffusion_name in switch.keys():
        return switch[diffusion_name]
    raise NotImplementedError