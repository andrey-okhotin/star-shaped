from data.Cifar10Dataset import Cifar10Dataset
from data.Text8Dataset import Text8Dataset
from data.DisastersDataset import DisastersDataset




def init_dataset(dataset_config):
    switch = {
        'cifar10'   : Cifar10Dataset,
        'text8'     : Text8Dataset,
        'disasters' : DisastersDataset
    }
    if dataset_config.name in switch:
        return switch[dataset_config.name](dataset_config)
    raise NotImplementedError