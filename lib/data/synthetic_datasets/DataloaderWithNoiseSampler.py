import torch
from torch.utils.data import Dataset, DataLoader
from copy import deepcopy




class SimpleDataset(Dataset):
    
    def __init__(
        self, 
        data_sampler, 
        num_iters, 
        batch_constructor, 
        diffusion,
        method,
        fixed_time
    ):
        self.data_sampler = data_sampler
        self.batch_constructor = batch_constructor
        self.diffusion = deepcopy(diffusion)
        self.num_iters = num_iters
        self.method = method
        self.fixed_time = fixed_time
        pass
    
    
    def __len__(self):
        return self.num_iters

    
    def __getitem__(self, idx):
        # get objects from dataset
        data = self.data_sampler.sample()
        if isinstance(data, tuple):
            x0, log_probs = data[0], data[1]
            batch = self.batch_constructor(x0)
            batch['log_probs'] = log_probs
        else:
            x0 = data
            batch = self.batch_constructor(x0)
            
        # sample time points
        if self.fixed_time is None:
            batch['t'] = self.diffusion.time_distribution.sample(batch)
        else:
            batch['t'] = torch.tensor(
                [ self.fixed_time ] * batch['batch_size'],
                device=batch['device']
            )
            
        # sample noise
        if   self.method == 'ss':
            batch['Gt'] = self.diffusion.sample_Gt(
                batch['x0'],
                batch['t']
            )
        elif self.method == 'ddpm':
            batch['xt'] = self.diffusion.forward_step_sample(
                batch['x0'],
                batch['t']
            )
        return batch
    

    
    
    
def custom_collate(original_batch):
    return original_batch[0]
    
    
    


def init_fast_dataloaders(
    dataset, 
    num_iters, 
    diffusion, 
    num_workers,
    method,
    fixed_time=None
):
    return (
        DataLoader(
            SimpleDataset(
                dataset.train_loader,
                num_iters,
                dataset.create_batch,
                diffusion,
                method,
                fixed_time
            ),
            num_workers=num_workers,
            collate_fn=custom_collate
        ),
        DataLoader(
            SimpleDataset(
                dataset.validation_loader,
                num_iters,
                dataset.create_batch,
                diffusion,
                method,
                fixed_time
            ),
            num_workers=num_workers,
            collate_fn=custom_collate
        )
    )
        
    
    