import torch
from torch.distributions.categorical import Categorical

from data.synthetic_datasets.distributions_utils import (
    get_unimodal_samples,
    get_multimodal_samples, 
    DistributionDataset
)





def u1_data(object_shape):
    unimodal_dataset = [
        Categorical(torch.tensor([0.10, 0.05, 0.70, 0.15])),
        Categorical(torch.tensor([0.24, 0.45, 0.14, 0.17])),
        Categorical(torch.tensor([0.03, 0.11, 0.26, 0.60])),
    ]
    num_categories = 4

    def sample(N, return_log_probs=False):
        token_i, log_probs_i = [], []
        for i, token_i_dist in enumerate(unimodal_dataset):
            one_hot = torch.zeros((N, num_categories))
            categorical_samples = token_i_dist.sample((N,))
            if return_log_probs:
                log_probs_i.append(token_i_dist.log_prob(categorical_samples))
            one_hot[torch.arange(N), categorical_samples] = 1
            token_i.append(one_hot)
        if return_log_probs:
            return (
                torch.stack(token_i).transpose(0, 1).reshape(-1, *object_shape),
                torch.stack(log_probs_i).transpose(0, 1)
            )
        return torch.stack(token_i).transpose(0, 1).reshape(-1, *object_shape)
    
    return sample






class CategoricalDataset(DistributionDataset):

    def __init__(self, categorical_config):
        self.object_shape = (1, 3, 4)
        switch = {
            'u1' : u1_data
        }
        self.sample_func = switch[categorical_config['dataset']](self.object_shape)
        super().__init__(
            categorical_config['size'],
            categorical_config['split'],
            categorical_config['batch_size']
        )
        pass