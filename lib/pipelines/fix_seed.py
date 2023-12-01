import torch
import random
import numpy
import tensorflow_probability




def fix_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    numpy.random.seed(seed)
    tensorflow_probability.random.sanitize_seed(seed)
    pass