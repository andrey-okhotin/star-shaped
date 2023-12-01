import math

import torch
import torch.nn as nn
from torch.nn.functional import softmax

from transformers import T5EncoderModel, T5Config




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




class T5EncoderDiffusionModel(nn.Module):
    
    def __init__(self, t5encoder_config):
        super().__init__()
        t5_config = T5Config()
        for k, v in t5encoder_config.items():
            t5_config.__dict__[k] = v
        t5_model = T5EncoderModel(t5_config)
        self.embed = t5_model.shared
        self.t5_encoder = t5_model.encoder
        self.linear_proj = nn.Linear(t5_config.d_model, t5_config.vocab_size)
        pass
    
    def forward(self, x, t):
        token_embeds = torch.matmul(x, self.embed.weight)
        seq_len, embed_sz = token_embeds.shape[-2:]
        pos_embeds = get_timestep_embedding(torch.arange(seq_len), embed_sz).to(x.device)
        seq_embeds = token_embeds + pos_embeds
        timestep_embeds = get_timestep_embedding(t, embed_sz).to(x.device)
        input_embeds = torch.concat((seq_embeds[:,0], timestep_embeds[:,None,:]), dim=1)
        output = self.t5_encoder(inputs_embeds=input_embeds)
        output_embeds = self.linear_proj(output.last_hidden_state[:,None,:-1,:])
        output_tokens_probs = softmax(output_embeds, dim=-1)
        return output_tokens_probs

