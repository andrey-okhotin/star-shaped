import torch




class TimeDistribution:
    
    """
    DESCRIPTION:
        Typical operations with time in diffusion models.
        
    """
    
    def __init__(self, num_steps):
        super().__init__()
        self.num_steps = num_steps
        pass
    
    
    def sample(self, batch, loss_object=None, mode=None):
        time_points = torch.randperm(self.num_steps)
        if time_points.shape[0] >= batch['batch_size']:
            time_points = time_points[:batch['batch_size']]
        else:
            duplicates = batch['batch_size'] // time_points.shape[0] + 1
            time_points = time_points.repeat(duplicates)[:batch['batch_size']]  
        return time_points.to(batch['device'])
    
    
    def get_time_points_tensor(self, batch, t):
        if t == -1:
            t = self.num_steps - 1
        return t * torch.ones(batch['batch_size'], dtype=torch.long, device=batch['device'])
    
    
    def reverse_time_iterator(self, batch, start_from=-1):
        if start_from == -1:
            start_from = self.num_steps - 1
        for t in range(start_from,-1,-1):
            yield t * torch.ones(batch['batch_size'], dtype=torch.long, device=batch['device']) 
    
    
    def forward_time_iterator(self, batch, start_from):
        for t in range(start_from,self.num_steps):
            yield t * torch.ones(batch['batch_size'], dtype=torch.long, device=batch['device'])
    
