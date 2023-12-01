import torch
from torch.distributions.dirichlet import Dirichlet
from torch.distributions.categorical import Categorical

from data.synthetic_datasets.distributions_utils import (
    get_unimodal_samples,
    get_multimodal_samples, 
    DistributionDataset
)





def u1_data(object_shape):
    unimodal_dataset = Dirichlet(torch.tensor([18, 1.2, 13.8]))
    
    def sample(N, return_log_probs=False):
        return get_unimodal_samples(
            unimodal_dataset,
            object_shape,
            num_samples=N,
            return_log_probs=return_log_probs
        )
    return sample





def m1_data(object_shape):
    multimodal_dataset = [
        Dirichlet(torch.tensor([28, 1.2, 16.8])),
        Dirichlet(torch.tensor([12, 12.2, 2.8])),
        Dirichlet(torch.tensor([6, 36.2, 6.8])),
    ]
    choice_probs = torch.tensor([0.3, 0.4, 0.4])
    choice = Categorical(choice_probs)
    
    def sample(N, return_log_probs=False):
        return get_multimodal_samples(
            choice,
            choice_probs,
            multimodal_dataset,
            object_shape,
            num_samples=N,
            return_log_probs=return_log_probs
        )        
    return sample





class DirichletDataset(DistributionDataset):

    def __init__(self, dirichlet_config):
        self.object_shape = (1, 1, 3)
        switch = {
            'u1' : u1_data,
            'm1' : m1_data
        }
        self.sample_func = switch[dirichlet_config['dataset']](self.object_shape)
        super().__init__(
            dirichlet_config['size'],
            dirichlet_config['split'],
            dirichlet_config['batch_size']
        )
        pass