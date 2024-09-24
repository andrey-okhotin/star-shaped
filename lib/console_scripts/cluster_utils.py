import os

import torch

from saving_utils.get_repo_root import get_repo_root




def save_config(config, rank):
    if rank == 0:
        if 'checkpoints' in config.keys():
            config_name = config.checkpoints.folder
        elif 'logs_file' in config.keys():
            config_name = config.logs_file
        json_path = os.path.join(get_repo_root(), 'saved_configs', config_name)
        with open(json_path + '.json', 'w') as f:
            f.write(config.to_json())
    pass




def logger_print(*args, **kwargs):
    if os.path.isabs(logger_print.file):
        path = logger_print.file
    else:
        path = os.path.join(get_repo_root(), '..', 'app', logger_print.file)
    with open(path, 'a') as f:
        for message in args:
            message = str(message) + " "
            f.write(message)
        if 'end' in kwargs.keys():
            f.write(kwargs['end'])
        else:
            f.write('\n')
    pass




def logger_reset():
    if os.path.isabs(logger_print.file):
        path = logger_print.file
    else:
        path = os.path.join(get_repo_root(), '..', 'app', logger_print.file)
    with open(path, 'w') as f:
        f.write('       ----------------------------\n')
        f.write('       |          LOGGER          |\n')
        f.write('       ----------------------------\n')
        f.write('\n\n')
    pass




def memory_reservation(
    gpu,
    print_func
):
    tensors = []
    c = 2**13
    while 1:
        try:
            tensors.append(
                torch.randn((c,c), dtype=torch.float32, device=torch.device(f'cuda:{gpu}'))
            )
        except:
            total_reserved_memory = torch.cuda.memory_reserved(gpu) / 2**30
            break
    for t in tensors:
        del t
    if not (print_func is None):
        print_func(f'gpu: {gpu} reserved memory: {total_reserved_memory:7.3f} Gb')
    return total_reserved_memory
