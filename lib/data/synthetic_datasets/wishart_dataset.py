import torch
from torch.distributions.wishart import Wishart
from torch.distributions.categorical import Categorical

from data.synthetic_datasets.distributions_utils import (
    get_unimodal_samples,
    get_multimodal_samples, 
    DistributionDataset
)





def u1_data(object_shape, device='cpu'):
    x0 = torch.tensor([[    2, -1.5 ],
                       [ -1.5,    2 ]], dtype=torch.float32).to(device)
    n = torch.tensor(400.).to(device)
    unimodal_dataset = torch.distributions.Wishart(n, x0 / n)
    
    def sample(N, return_log_probs=False):
        return get_unimodal_samples(
            unimodal_dataset,
            object_shape,
            num_samples=N,
            return_log_probs=return_log_probs
        )
    return sample





def m1_data(object_shape, device='cpu'):
    k = [ 
        torch.tensor(320.), 
        torch.tensor(40.), 
        torch.tensor(120.) 
    ]
    V = [
        torch.tensor([[10.,-9], [-9,10]]) / (k[0] * 540/400),
        torch.tensor([[12., 3], [ 3, 4]]) / (k[1] * 440/300),
        torch.tensor([[ 6., 8], [ 8,25]]) / (k[2] * 720/200)
    ]
    clusters_expectations = torch.stack(V) * torch.tensor(k)[:,None,None]
    multimodal_dataset = [ 
        Wishart(k[0].to(device), V[0].to(device)), 
        Wishart(k[1].to(device), V[1].to(device)), 
        Wishart(k[2].to(device), V[2].to(device)) 
    ]
    choice_probs = [ 
        torch.tensor(0.3), 
        torch.tensor(0.4), 
        torch.tensor(0.4)
    ]
    choice = Categorical(torch.tensor(choice_probs).to(device))

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





class WishartDataset(DistributionDataset):

    def __init__(self, wishart_config):
        self.object_shape = (1, 2, 2)
        switch = {
            'u1' : u1_data,
            'm1' : m1_data
        }
        if 'device' in wishart_config.keys():
            device = wishart_config['device']
        else:
            device = 'cpu'
        self.sample_func = switch[wishart_config['dataset']](
            self.object_shape, device
        )
        super().__init__(
            wishart_config['size'],
            wishart_config['split'],
            wishart_config['batch_size']
        )
        pass