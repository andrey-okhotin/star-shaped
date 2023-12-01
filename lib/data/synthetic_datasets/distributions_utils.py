import torch




class DataSampler:
    
    def __init__(self, bs, size, sample_func=None, dataset=(None,None)):
        self.bs = bs
        self.size = size
        self.return_log_probs = False
        self.sample_func = sample_func
        self.dataset, self.log_probs = dataset[0], dataset[1]
        self.index = 0
        pass
    
    
    def sample(self):
        while 1:
            if self.size == 'inf':
                return self.sample_func(self.bs, self.return_log_probs)
            else:
                if self.index == 0:
                    rand_index = torch.randperm(self.size)
                    self.dataset = self.dataset[rand_index]
                    self.log_probs = self.log_probs[rand_index]
                self.index = (self.index + 1) % max(self.size//self.bs, 1)
                if self.return_log_probs:
                    return (
                        self.dataset[self.index*self.bs:(self.index+1)*self.bs],
                        self.log_probs[self.index*self.bs:(self.index+1)*self.bs]
                    )
                else:
                    return self.dataset[self.index*self.bs:(self.index+1)*self.bs]
        pass
    
    
    def true_sample(self):
        return self.sample_func(self.bs, self.return_log_probs)
    
    
    def __iter__(self):
        self.index = 0
        while 1:
            if self.size == 'inf':
                yield self.sample_func(self.bs, self.return_log_probs)
            else:
                if self.index == 0:
                    rand_index = torch.randperm(self.size)
                    self.dataset = self.dataset[rand_index]
                    self.log_probs = self.log_probs[rand_index]
                self.index = (self.index + 1) % max(self.size//self.bs, 1)
                if self.return_log_probs:
                    yield (
                        self.dataset[self.index*self.bs:(self.index+1)*self.bs],
                        self.log_probs[self.index*self.bs:(self.index+1)*self.bs]
                    )
                else:
                    yield self.dataset[self.index*self.bs:(self.index+1)*self.bs]
        pass




class DistributionDataset:

    def __init__(self, size, split, batch_size):
        self.size = size
        if size == 'inf':
            self.train_dataloader = self.validation_dataloader = DataSampler(
                batch_size, size, sample_func=self.sample_func
            )
            
        else:
            train_size = round(split * size)
            self.train_dataset, self.train_log_probs = [], []
            for i in range(train_size // batch_size + 1):
                samples, log_probs = self.sample_func(batch_size, return_log_probs=True)
                self.train_dataset.append(samples)
                self.train_log_probs.append(log_probs)
            self.train_dataset = torch.vstack(self.train_dataset)[:train_size]
            self.train_log_probs = torch.hstack(self.train_log_probs)[:train_size]
            self.train_dataloader = DataSampler(
                batch_size, train_size, 
                sample_func=self.sample_func,
                dataset=(self.train_dataset, self.train_log_probs)
            )
            
            val_size = round((1 - split) * size)
            self.validation_dataset, self.validation_log_probs = [], []
            for i in range(val_size // batch_size + 1):
                samples, log_probs = self.sample_func(batch_size, return_log_probs=True)
                self.validation_dataset.append(samples)
                self.validation_log_probs.append(log_probs)
            self.validation_dataset = torch.vstack(self.validation_dataset)[:val_size]
            self.validation_log_probs = torch.hstack(self.validation_log_probs)[:val_size]
            self.validation_dataloader = DataSampler(
                batch_size, val_size,
                sample_func=self.sample_func,
                dataset=(self.validation_dataset, self.validation_log_probs)
            )
        pass


    def sample(self, loader_type, sample_size, true_sample=False):
        if loader_type == 'train':
            dataloader = self.train_dataloader
            bs = self.train_dataloader.bs
        else:
            dataloader = self.validation_dataloader
            bs = self.validation_dataloader.bs
            
        samples, log_probs = [], []
        for i in range(sample_size // bs + 1):
            if true_sample:
                x0 = dataloader.true_sample()
            else:
                x0 = dataloader.sample()
            if isinstance(x0, tuple):
                samples.append(x0[0])
                log_probs.append(x0[1])
            else:
                samples.append(x0)
            
        if len(log_probs) > 0:
            return (
                torch.vstack(samples)[:sample_size],
                torch.hstack(log_probs)[:sample_size]
            )
        return torch.vstack(samples)[:sample_size]


    def set_batch_size(self, batch_size):
        self.train_dataloader.bs = batch_size
        self.validation_dataloader.bs = batch_size
        pass
    
    
    def set_return_log_probs(self, bool_flag):
        self.train_dataloader.return_log_probs = bool_flag
        self.validation_dataloader.return_log_probs = bool_flag
        pass
    
    


        
def get_unimodal_samples(
    unimodal_dataset,
    object_shape,
    num_samples,
    return_log_probs=False
):
    output_shape = (num_samples, *object_shape)
    samples = unimodal_dataset.sample((num_samples,))
    if return_log_probs:
        log_probs = unimodal_dataset.log_prob(samples)
        log_probs = log_probs.reshape(num_samples, -1).sum(dim=1)
        return samples.reshape(output_shape), log_probs
    return samples.reshape(output_shape)
        
        
        
        

def get_multimodal_samples(
    choice,
    choice_probs,
    multimodal_dataset, 
    object_shape,
    num_samples, 
    return_log_probs=False
):
    output_shape = (num_samples, *object_shape)
    choices = choice.sample((num_samples,))
    samples = []
    log_probs = []
    for i in range(len(choice_probs)):
        num_i_samples = (choices == i).sum().item()
        samples.append(multimodal_dataset[i].sample((num_i_samples,)))
        if return_log_probs:
            cluster_log_probs = torch.log(torch.tensor([ choice_probs[i] ] * num_i_samples))
            log_p = multimodal_dataset[i].log_prob(samples[-1]
                                               ).reshape(num_i_samples, -1).sum(dim=1)
            log_probs.append(cluster_log_probs + log_p)
    if return_log_probs:
        return torch.vstack(samples).reshape(output_shape), torch.hstack(log_probs)
    return torch.vstack(samples).reshape(output_shape)