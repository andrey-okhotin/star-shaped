import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

import os
import shutil
import zipfile
import pickle
import urllib.request

from saving_utils.get_repo_root import get_repo_root


class Text8Dataset:
    
    def __init__(self, dataset_config):
        base_datasets_root = os.path.join(get_repo_root(), 'datasets')
        def get_dataset_path(path):
            return os.path.join(base_datasets_root, path)

        url = 'http://mattmahoney.net/dc/text8.zip'
        filename = get_dataset_path('text8.zip')
        if not os.path.isfile(filename):
            print('Downloading text8 dataset...')
            with urllib.request.urlopen(url) as response, \
                open(filename, 'wb') as outfile:
                shutil.copyfileobj(response, outfile)
        rawdata = zipfile.ZipFile(filename).read('text8').decode('utf-8')
        
        self.vocab, self.inverse_vocab = {}, {}
        index = 0
        for i, token in enumerate(rawdata):
            if not (token in self.vocab):
                self.vocab[token] = index
                self.inverse_vocab[index] = token
                index += 1
        pad_index = self.vocab[' ']
        
        split = ( (0.00, 0.90), (0.90, 0.95), (0.95, 1.00) )
        data_size = len(rawdata)
        self.train, self.val, self.test = [], [], []
        for c in rawdata[int(split[0][0]*data_size):int(split[0][1]*data_size)]:
            self.train.append(self.vocab[c])
        for c in rawdata[int(split[1][0]*data_size):int(split[1][1]*data_size)]:
            self.val.append(self.vocab[c])
        for c in rawdata[int(split[2][0]*data_size):int(split[2][1]*data_size)]:
            self.test.append(self.vocab[c])
            
        self.dataset_config = dataset_config
        self.batch_size = self.dataset_config['batch_size']
        self.train_loader = DataLoader(
            dataset=CharacterLevelDataset(
                dataset=self.train, 
                seq_len=self.dataset_config['seq_len'],
                random_crop=True
            ),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.dataset_config['num_workers']
        )
        self.validation_loader = DataLoader(
            dataset=CharacterLevelDataset(
                dataset=self.val, 
                seq_len=self.dataset_config['seq_len'],
                random_crop=False
            ),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.dataset_config['num_workers']
        )
        self.test_loader = DataLoader(
            dataset=CharacterLevelDataset(
                dataset=self.test,
                seq_len=self.dataset_config['seq_len'],
                random_crop=False
            ),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.dataset_config['num_workers']
        )
        self.num_batches = self.train_loader.dataset.dataset_size // self.batch_size
        pass
    
    
    def back_transform(self, tokens, temperature=None):
        # tokens.shape = (bs, 1, seq_len, token_probs)
        str_list = []
        for i1 in range(len(tokens)):
            str_list.append("")
            for i2 in range(len(tokens[i1,0])):
                if temperature is None:
                    token = tokens[i1,0,i2].argmax().item()
                else:
                    raise NotImplementedError
                str_list[-1] = str_list[-1] + self.inverse_vocab[token]
        return str_list
    
    
    def create_batch(self, x0, device='cpu'):
        if device == 'cuda':
            x0 = x0.cuda(non_blocking=True)
        return {
            'x0' : x0,
            'batch_size' : x0.shape[0],
            'device' : x0.device
        }


    def save_generated_objects(self, objects, folder, rank):
        texts = self.back_transform(objects)
        already_sampled = len(os.listdir(folder))
        for i in range(len(texts)):
            index = already_sampled + i
            num = '0' * (5 - len(str(index))) + str(index)
            file_name = os.path.join(folder, f'str_{rank}_{num}.txt')
            strsave(file_name, texts[i])
        pass
    
    
    def to_distributed(self, dataset, rank, world_size):
        if dataset == 'train':
            data_loader = self.train_loader
        elif dataset == 'validation':
            data_loader = self.validation_loader
        elif dataset == 'test':
            data_loader = self.test_loader
        
        self.distributed_sampler = DistributedSampler(
            data_loader.dataset,
            num_replicas=world_size,
            rank=rank
        )
        self.dist_bs = self.batch_size // world_size + 1
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
        self.train_loader = DataLoader(
            dataset=self.train_loader.dataset,
            batch_size=self.batch_size,
            shuffle=True
        )
        self.validation_loader = DataLoader(
            dataset=self.validation_loader.dataset,
            batch_size=self.batch_size,
            shuffle=False
        )
        self.test_loader = DataLoader(
            dataset=self.test_loader.dataset,
            batch_size=self.batch_size,
            shuffle=False
        )
        pass

        
    
    
class CharacterLevelDataset(Dataset):
    
    def __init__(self, dataset, seq_len, random_crop):
        self.dataset = torch.tensor(dataset)
        self.seq_len = seq_len
        self.random_crop = random_crop
        if random_crop:
            self.seq_len *= 2
        end_index = self.dataset.shape[0] % self.seq_len
        self.cur_seq_split = self.dataset[:self.dataset.shape[0]-end_index]
        self.cur_seq_split = self.cur_seq_split.reshape(-1, self.seq_len)
        self.dataset_size = self.cur_seq_split.shape[0]
        if random_crop:
            self.seq_len //= 2
        pass
    
    def __len__(self):
        return self.cur_seq_split.shape[0]
    
    def __getitem__(self, i):
        if self.random_crop:
            crop_start = torch.randint(0,self.seq_len,(1,)).item()
            x0 = self.cur_seq_split[i,crop_start:crop_start+self.seq_len].clone()
        else:
            x0 = self.cur_seq_split[i].clone()
            
        one_hot = torch.zeros((1,x0.shape[0],27), dtype=torch.float32)
        one_hot[0,torch.arange(x0.shape[0]),x0] = 1
        return one_hot




def strsave(fpath, s):
    with open(fpath, 'w') as f:
        f.write(s)


