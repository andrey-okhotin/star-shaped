import os

import torch
import torch.distributed as dist

from data.Datasets import init_dataset
from diffusion.Diffusion import init_diffusion
from models.Models import init_model

from saving_utils.get_repo_root import get_repo_root


    
    
class UniversalNLLEstimator:
    
    def __init__(self, config):
        self.config = config
        self.data_generator = init_dataset(config.data)
        self.model = init_model(config.model)
        self.diffusion = init_diffusion(config.diffusion)
        
        # in general case you need to set dequantizator p(x_0|G_1)
        # this case not implemented in this repo
        if 'dequantizator' in config:
            dequantizator = init_dequantizator(config.dequantizator)
            self.diffusion.set_dequantizator(dequantizator)
        
        nll_results_dir = os.path.join(get_repo_root(), 'results', 'nll_estimations')
        if process.is_root_process and not os.path.exists(nll_results_dir):
            os.mkdir(nll_results_dir)
        else:
            time.sleep(360)
        self.save_file = os.path.join(nll_results_dir, config.save_folder)
        pass

        
    def estimate(
        self,
        num_samples,
        dataset_part,
        num_iwae_trajectories,
        process,
        fprint=print
    ):
        torch.cuda.set_device(process.gpu)
        self.model.cuda(process.gpu)
        self.diffusion.cuda(process.gpu)

        new_batch_size = 64
        prev_batch_size = self.data_generator.batch_size
        self.data_generator.change_batch_size_in_dataloaders(new_batch_size)
        if   hasattr(self.diffusion, 'precompute_tail_normalization_statistics'):
            self.diffusion.precompute_tail_normalization_statistics(self.data_generator, 2000)
        elif hasattr(self.diffusion, 'precompute_xt_normalization_statistics'):
            self.diffusion.precompute_xt_normalization_statistics(self.data_generator, 2000)
        self.data_generator.change_batch_size_in_dataloaders(prev_batch_size)
        
        if process.distributed:
            num_samples = (
                (num_samples // process.world_size) + (num_samples % process.world_size > 0))
            self.data_generator.to_distributed(dataset_part, process.rank, process.world_size)
            self.data_generator.change_batch_size_in_dataloaders(self.data_generator.dist_bs)
            fprint( f'proc: {process.rank} bs: {self.data_generator.dist_bs}', flush=True)
            data_loader = self.data_generator.distributed_loader
        else:
            if dataset_part == 'train':
                data_loader = self.data_generator.train_loader
            elif dataset_part == 'validation':
                data_loader = self.data_generator.validation_loader
            elif dataset_part == 'test':
                data_loader = self.data_generator.test_loader
        
        iwae_log_probs = torch.zeros((0,), dtype=torch.float32)
        K = torch.tensor(num_iwae_trajectories)
        self.model.eval()
        with torch.no_grad():
            for batch_index, x0 in enumerate(data_loader):
                x0 = x0.repeat(K, 1, 1, 1)
                batch = self.data_generator.create_batch(x0, device='cuda')
                ll = self.diffusion.ll_estimation(
                    batch=batch, 
                    model=self.model
                )
                L = torch.logsumexp(ll.reshape(K, -1), dim=0) - torch.log(K)
                iwae_log_probs = torch.hstack((iwae_log_probs, L))
                torch.save(iwae_log_probs, f"{self.save_file}_p{process.rank}.pt")
                fprint(f'proc: {process.rank} batch: {batch_index}', flush=True)
                if iwae_log_probs.shape[0] >= num_samples:
                    break

        if process.distributed:
            dist.barrier()
        
        if process.is_root_process:
            fprint(f'concatinate from:', flush=True)
            iwae_from_all_processes = []
            for p in range(process.world_size):
                f = f'{self.save_file}_p{p}.pt'
                fprint(f'    {f}', flush=True)
                iwae_from_all_processes.append(torch.load(f))
                os.remove(f)
            ll_per_object = torch.hstack(iwae_from_all_processes)
            nll_per_object = -ll_per_object / torch.log(torch.tensor(2.))
            nll = nll_per_object.mean()
            torch.save(nll_per_object, f"{self.save_file}_per_object.pt")
            torch.save(nll, f"{self.save_file}.pt")
            fprint(f'NLL: {nll.item():8.3f}', flush=True)
                
        self.model.to('cpu')
        self.model.device = 'cpu'
        self.diffusion.to('cpu')
        return iwae_log_probs