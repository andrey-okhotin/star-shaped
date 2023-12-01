import os
import numpy as np
from matplotlib import pyplot as plt
from skimage.io import imsave

import torch
from torchvision.utils import save_image
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import CIFAR10
from torchvision.transforms import (
    Resize,
    Compose,
    ToTensor,
    RandomHorizontalFlip,
    Normalize
)

from saving_utils.get_repo_root import get_repo_root



class Cifar10Dataset:
    
    def __init__(self, dataset_config):
        self.dataset_config = dataset_config
        self.batch_size = dataset_config['batch_size']
        self.num_workers = dataset_config['num_workers']
        self.batch_shape = (self.batch_size, 3, 32, 32)
        self.create_dataloaders(self.batch_size)
        self.num_objects = len(self.train_loader.dataset)
        self.num_batches = ( self.num_objects // self.batch_size + 
                             (self.num_objects % self.batch_size > 0) )
        pass


    def create_dataloaders(self, batch_size):
        base_datasets_root = os.path.join(get_repo_root(), 'datasets')
        train_transforms = Compose([
            Resize((32, 32)),
            RandomHorizontalFlip(p=0.5),
            ToTensor(),
            Normalize(mean=0.5, std=0.5)
        ])
        validation_transforms = Compose([
            Resize((32, 32)),
            ToTensor(),
            Normalize(mean=0.5, std=0.5)
        ])
        self.train_loader = DataLoader(
            CIFAR10(root='/opt/software/datasets/cifar/', download=False, 
                    train=True, transform=train_transforms),
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers
        )
        self.validation_loader = DataLoader(
            CIFAR10(root='/opt/software/datasets/cifar/', download=False, 
                    train=False, transform=validation_transforms),
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers
        )
        pass
    
    
    def create_batch(self, x0, device='cpu', info=None):
        if   device == 'cpu':
            x0 = x0[0]
        elif device == 'cuda':
            x0 = x0[0].cuda(non_blocking=True)
        return {
            'x0'         : x0,
            'channels'   : x0.shape[1],
            'resolution' : x0.shape[2],
            'batch_size' : x0.shape[0],
            'device'     : x0.device
        }
    
    
    def back_transform(self, image_batch):
        if isinstance(image_batch, torch.Tensor):
            single_image = False
            if len(image_batch.shape) == 3:
                single_image = True
                image_batch = image_batch.view(1,*image_batch.shape)
            inverse_scaler = lambda x: torch.clip(127.5 * (x + 1), 0, 255)
            image_batch = inverse_scaler(image_batch).permute(0,2,3,1).data.numpy().astype(np.uint8)
            if single_image:
                image_batch = image_batch[0]
        else:
            raise TypeError
        return image_batch

    
    def save_generated_objects(self, objects, folder, rank):
        images = self.back_transform(objects)
        already_sampled = len(os.listdir(folder))
        for i in range(images.shape[0]):
            index = already_sampled + i
            num = '0' * (5 - len(str(index))) + str(index)
            file_name = os.path.join(folder, f'img_{rank}_{num}.png')
            imsave(file_name, images[i])
        pass
    
    
    def to_distributed(self, dataset, rank, world_size):
        if dataset == 'train':
            data_loader = self.train_loader
        elif dataset == 'validation':
            data_loader = self.validation_loader

        self.distributed_sampler = DistributedSampler(
            data_loader.dataset,
            num_replicas=world_size,
            rank=rank
        )
        self.dist_bs = self.dataset_config['batch_size'] // world_size + 1
        self.distributed_loader = DataLoader(
            dataset=data_loader.dataset,
            batch_size=self.dist_bs,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            sampler=self.distributed_sampler
        )
        pass
    
    
    def change_batch_size_in_dataloaders(self, batch_size):
        self.batch_size = batch_size
        self.create_dataloaders(self.batch_size)
        pass


    def plot_image_from_dataset(self, idx, set_name):
        if set_name == 'train':
            img_tensor = self.train_loader.dataset[idx][0]
        if set_name == 'validation':
            img_tensor = self.validation_loader.dataset[idx][0]
        fig = plt.figure(figsize=(6,6))
        fig.patch.set_facecolor('white')
        plt.imshow(self.back_transform(img_tensor))
        plt.axis('off')
        plt.show()
        pass
    
    
    def extract_dataset_to_folder(self, folder, mode):
        folder = os.path.join(get_repo_root(), 'datasets', folder)
        if not os.path.exists(folder):
            os.mkdir(folder)
        if mode == 'train':
            loader = self.train_loader
        else:
            loader = self.validation_loader 
        already_extracted = 0
        for x0 in loader:
            batch = self.create_batch(x0)
            imgs = self.back_transform(batch['x0'])
            for img_index in range(imgs.shape[0]):
                img_num = already_extracted + img_index
                num = '0' * (5 - len(str(img_num))) + str(img_num)
                file_name = os.path.join(folder, f'img_{num}.png')
                save_image(imgs[img_index], file_name)
            already_extracted += imgs.shape[0]
        pass