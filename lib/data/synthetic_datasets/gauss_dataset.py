import torch
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical

from data.synthetic_datasets.distributions_utils import (
    get_unimodal_samples,
    get_multimodal_samples, 
    DistributionDataset
)





def u1_data(object_shape):
    unimodal_dataset = Normal(
        torch.tensor([0.25, -0.1, -0.5]),
        torch.tensor([0.5, 2., 0.3])
    )
    
    def sample(N, return_log_probs=False):
        return get_unimodal_samples(
            unimodal_dataset,
            object_shape,
            num_samples=N,
            return_log_probs=return_log_probs
        )
    return sample








class GaussDataset(DistributionDataset):

    def __init__(self, gauss_config):
        self.object_shape = (1, 1, 3)
        switch = {
            'u1' : u1_data
        }
        self.sample_func = switch[gauss_config['dataset']](self.object_shape)
        super().__init__(
            gauss_config['size'],
            gauss_config['split'],
            gauss_config['batch_size']
        )
        pass