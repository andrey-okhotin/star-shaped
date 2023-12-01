import torch
import torch.nn as nn

import math




def get_timestep_embedding(timesteps, embedding_dim, base=10000):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1
    half_dim = embedding_dim // 2
    emb = math.log(base) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb




def get_activation(nonlinearity):
    """Get activation functions from the config file."""
    if   nonlinearity.lower() == 'elu':
        return nn.ELU()
    elif nonlinearity.lower() == 'relu':
        return nn.ReLU()
    elif nonlinearity.lower() == 'lrelu':
        return nn.LeakyReLU(negative_slope=0.2)
    elif nonlinearity.lower() == 'swish':
        return nn.SiLU()
    else:
        raise NotImplementedError('activation function does not exist!')




def domain_transform_positive_definite_matrices_2x2(x, x_shape):
    bs = x.shape[0]
    zeros = torch.zeros((bs,1), dtype=x.dtype, device=x.device)
    L = torch.hstack((x, zeros))[:,[0,3,1,2]].reshape(bs,2,2)
    V = torch.matmul(L, L.transpose(1,2))
    eps = 1e-4
    I = torch.eye(2, device=x.device)[None,:,:].repeat(bs,1,1)
    return (V + eps * I).reshape(x_shape)




def domain_transform_simplex(x, x_shape):
    return nn.functional.softmax(x, dim=1).reshape(x_shape)




def domain_transform_sphere(x, x_shape):
    return (x / torch.norm(x, dim=-1, keepdim=True)).reshape(x_shape)




def domain_transformation(x, domain, x_shape):
    switch = {
        'pdm_2x2' : domain_transform_positive_definite_matrices_2x2,
        'simplex' : domain_transform_simplex,
        'sphere'  : domain_transform_sphere
    }
    return switch[domain](x, x_shape)


    

class MLP(nn.Module):
    
    def __init__(self, model_config):
        super().__init__()
        self.time_embed_dim = model_config.time_embed_dim
        self.domain = model_config.domain
        self.encode = nn.Sequential(
            nn.Linear(model_config.input_dim + self.time_embed_dim, model_config.hidden_dims[0]),
            get_activation(model_config.act)
        )
        self.layers = []
        for hidden_dim in model_config.hidden_dims:
            self.layers.append(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim), 
                get_activation(model_config.act)
            ))
            
        self.layers = nn.Sequential(*self.layers)
        self.decode = nn.Linear(model_config.hidden_dims[-1], model_config.output_dim)
        pass
    
    
    def forward(self, x, t, t_index=None):
        x_shape = x.shape
        bs = x.shape[0]
        h = x.reshape(bs,-1)
        time_embeds = get_timestep_embedding(t, self.time_embed_dim, base=10000)
        input = torch.hstack((h, time_embeds))
        h = self.encode(input)
        for hidden_layer in self.layers:
            h = h + hidden_layer(h)
        x_pred = self.decode(h)
        return domain_transformation(x_pred, self.domain, x_shape)