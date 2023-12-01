import os
import pickle

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader

from pipelines.fix_seed import fix_seed
from saving_utils.get_repo_root import get_repo_root




class DisastersDataset:

    def __init__(self, dataset_config):
        self.dataset_config = dataset_config
        self.batch_size = dataset_config['batch_size']
        self.num_workers = dataset_config['num_workers']
        self.dataset_seed = dataset_config['split_seed']
        switch = {
            'fire'       : (FireDataset,       'fire.csv'      ),
            'flood'      : (FloodDataset,      'flood.csv'     ),
            'earthquake' : (EarthQuakeDataset, 'earthquake.csv'),
            'volcano'    : (VolcanoDataset,    'volcano.csv'   )
        }
        if dataset_config.disaster in switch:
            data_class, data_file = switch[dataset_config.disaster]
            dataset_path = os.path.join(get_repo_root(), 'datasets', data_file)
            dataset = data_class(dataset_path)
        else:
            raise NotImplementedError

        train_d, validation_d, test_d = random_train_val_test_split(
            dataset,
            val_size=0.1,
            test_size=0.1,
            seed=self.dataset_seed,
            save_split=dataset_config['save_split']
        )
        self.train_dataset = train_d
        self.validation_dataset = validation_d
        self.test_dataset = test_d
        
        self.num_batches = len(self.train_dataset) // self.batch_size                    
        self.create_dataloaders()
        pass


    def create_dataloaders(self):
        self.train_loader = DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers
        )
        self.validation_loader = DataLoader(
            self.validation_dataset, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers
        )
        self.test_loader = DataLoader(
            self.test_dataset, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers
        )
        pass

    
    def back_transform(self, x):
        x = x / torch.norm(x, dim=-1, keepdim=True)
        angle = torch.zeros(*x.shape[:-1], 2)
        angle[..., 0] = torch.atan2(x[..., 1], x[..., 0])
        angle[..., 1] = torch.arcsin(x[..., 2])
        return (180 * angle / np.pi).reshape(-1,2)

    
    def create_batch(self, x0, device='cpu'):
        if   device == 'cpu':
            x0 = x0
        elif device == 'cuda':
            x0 = x0.cuda(non_blocking=True)
        return {
            'x0'         : x0,
            'batch_size' : x0.shape[0],
            'device'     : x0.device
        }

    
    def save_generated_objects(self, objects, folder, rank):
        already_sampled = len(os.listdir(folder))
        torch.save(
            torch.tensor(objects),
            os.path.join(folder, f'r{rank}_coords{already_sampled}.pt')
        )
        pass


    def change_batch_size_in_dataloaders(self, batch_size):
        self.batch_size = batch_size
        self.create_dataloaders()
        pass


    def sample(self, loader_type, sample_size):
        switch = {
            'train'      : self.train_dataset,
            'validation' : self.validation_dataset,
            'test'       : self.test_dataset
        }
        dataset = switch[loader_type]
        return dataset.__getitem__(np.arange(min(sample_size, len(dataset))))




def random_train_val_test_split(dataset, val_size, test_size, seed=0, save_split=None):
    # generate split
    fix_seed(seed)
    n_samples = len(dataset)
    train_size = 1 - val_size - test_size
    train_idx = np.random.choice(range(int(n_samples)), int(n_samples * train_size), replace=False)
    remaining = set(range(n_samples)) - set(train_idx)
    val_idx = np.random.choice(list(remaining), int(n_samples * val_size),  replace=False)
    test_idx = list(set(remaining) - set(val_idx))

    if not (save_split is None):
        indices_dict = {
            'train_idx' : train_idx, 
            'val_idx'   : val_idx, 
            'test_idx'  : test_idx
        }
        splits_folder = os.path.join(get_repo_root(), 'results', save_split)
        if not os.path.exists(splits_folder):
            os.mkdir(splits_folder)
        save_file = os.path.join(splits_folder, f'indices_{dataset.name}_{seed}.pkl')
        with open(save_file, 'wb') as handle:
            pickle.dump(indices_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return (
        dataset.indices_slice(train_idx), 
        dataset.indices_slice(val_idx), 
        dataset.indices_slice(test_idx)
    )




class DisasterDatasetClass(Dataset):

    def __init__(self):
        lat = np.pi * torch.tensor([self.lats], dtype=torch.float32) / 180.
        lon = np.pi * torch.tensor([self.lons], dtype=torch.float32) / 180.
        self.coords = torch.cat([
            torch.cos(lat) * torch.cos(lon),
            torch.cos(lat) * torch.sin(lon),
            torch.sin(lat)
        ]).T.reshape(-1,1,1,3)
        pass
      
    def __getitem__(self, idx):
        return self.coords[idx]
   
    def indices_slice(self, idx):
        return type(self)(self.file_path, idx=idx)
    
    def __len__(self):
        return self.coords.shape[0]
    



class FireDataset(DisasterDatasetClass):
    
    def __init__(self, file_path, idx=None):
        self.name = 'fire'
        self.file_path = file_path
        self.data = pd.read_csv(file_path)
        if idx is None:
            idx = range(len(self.data))
        self.data = self.data.iloc[idx]
        self.data['lat'] = self.data.index
        self.data['lon'] = self.data['# https://earthdata.nasa.gov/earth-observation-data/near-real-time/firms/active-fire-data']
        self.lons = self.data['lon'].values
        self.lats = self.data['lat'].values
        super().__init__()
        pass




class FloodDataset(DisasterDatasetClass):
    
    def __init__(self, file_path, idx=None):
        self.name = 'flood'
        self.file_path = file_path
        self.data = pd.read_csv(file_path).iloc[1:]
        if idx is None:
            idx = list(range(len(self.data)))
        self.data = self.data.iloc[idx]
        self.data['lat'] = self.data.index
        self.data['lon'] = self.data['# http://floodobservatory.colorado.edu/Archives/index.html']
        self.lons = self.data['lon'].astype(float).values
        self.lats = self.data['lat'].astype(float).values
        super().__init__()
        pass




class EarthQuakeDataset(DisasterDatasetClass):
    
    def __init__(self, file_path, idx=None):
        self.name = 'earthquake'
        self.file_path = file_path
        self.data = pd.read_csv(file_path, skiprows=[0, 1, 2]).iloc[1:]
        if idx is None:
            idx = list(range(len(self.data)))
        self.data = self.data.iloc[idx]
        self.data['lat'] = self.data.LATITUDE
        self.data['lon'] = self.data.LONGITUDE
        self.lons = self.data['lon'].astype(float).values
        self.lats = self.data['lat'].astype(float).values
        super().__init__()
        pass



    
class VolcanoDataset(DisasterDatasetClass):
    def __init__(self, file_path, idx=None):
        self.name = 'volcano'
        self.file_path = file_path
        self.data = pd.read_csv(file_path, skiprows=[0]).iloc[1:]
        if idx is None:
            idx = list(range(len(self.data)))
        self.data = self.data.iloc[idx]
        self.data['lat'] = self.data.Latitude
        self.data['lon'] = self.data.Longitude
        self.lons = self.data['lon'].astype(float).values
        self.lats = self.data['lat'].astype(float).values
        super().__init__()
        pass




