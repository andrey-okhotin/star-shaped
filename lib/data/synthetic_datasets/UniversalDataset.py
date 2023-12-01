import torch

from data.synthetic_datasets.dirichlet_dataset import DirichletDataset
from data.synthetic_datasets.wishart_dataset import WishartDataset
from data.synthetic_datasets.gauss_dataset import GaussDataset
from data.synthetic_datasets.categorical_dataset import CategoricalDataset





class UniversalDataset:

    def __init__(self, dataset_config):
        self.dataset_config = dataset_config
        self.switch = {
            'simplex'     : DirichletDataset,
            'pdm_2x2'     : WishartDataset,
            'real_num'    : GaussDataset,
            'categorical' : CategoricalDataset
        }
        self.dataset_type = dataset_config['type']
        self.dataset_object = self.switch[dataset_config['type']](dataset_config)
        self.batch_size = dataset_config['batch_size']
        self.train_loader = self.dataset_object.train_dataloader
        self.validation_loader = self.dataset_object.validation_dataloader
        self.object_shape = self.dataset_object.object_shape
        self.num_objects = self.dataset_object.size
        self.sample = self.dataset_object.sample
        pass


    def create_batch(self, x0, info=None):
        return {
            'x0' : x0,
            'shape' : self.object_shape,
            'batch_size' : self.batch_size,
            'device' : x0.device,
            'info' : info
        }
    
    
    def batch_to(self, batch, device):
        for k in batch.keys():
            if isinstance(batch[k], torch.Tensor):
                batch[k] = batch[k].to(device)
        batch['device'] = device
        pass

    
    def set_batch_size(self, batch_size):
        self.batch_size = batch_size
        self.dataset_object.set_batch_size(batch_size)
        pass
    
    
    def set_return_log_probs(self, bool_flag):
        self.dataset_object.set_return_log_probs(bool_flag)


    
    
    
def init_universal_dataset(dataset_config):
    return UniversalDataset(dataset_config)