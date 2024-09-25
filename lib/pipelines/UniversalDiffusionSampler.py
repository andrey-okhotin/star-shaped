import os
import time
import shutil

import torch

from saving_utils.get_repo_root import get_repo_root
from visualization_utils.ProgressPrinters import steps_printer

from data.Datasets import init_dataset
from diffusion.Diffusion import init_diffusion
from models.Models import init_model




class UniversalDiffusionSampler:

    def __init__(self, config, process):
        self.config = config
        self.data_generator = init_dataset(config.data)
        self.diffusion = init_diffusion(config.diffusion)
        self.model = init_model(config.model)

        if os.path.isabs(config.save_folder):
            config.save_folder = os.path.basename(os.path.normpath(config.save_folder))
        main_folder = os.path.join(get_repo_root(), '..', 'app', config.save_folder)
        self.folder = os.path.join(main_folder, 'generated_samples')        
        if process.distributed:
            if process.is_root_process:
                if os.path.exists(main_folder):
                    shutil.rmtree(main_folder)
                os.mkdir(main_folder)
                os.mkdir(self.folder)
            else:
                time.sleep(60)
        else:
            if os.path.exists(main_folder):
                shutil.rmtree(main_folder)
            os.mkdir(main_folder)
            os.mkdir(self.folder)
        os.chmod(main_folder, 0o777)
        os.chmod(self.folder, 0o777)
        pass


    

    def sample(
        self, 
        num_samples,
        process,
        fprint=print
    ):
        # process and gpu coordination
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
        
        num_iterations = (num_samples // self.data_generator.batch_size +
                          (num_samples % self.data_generator.batch_size > 0))
        last_batch_size = num_samples % self.data_generator.batch_size
        
        self.model.eval()
        with torch.no_grad():
            torch.cuda.empty_cache()
            
            for batch_index, x0 in enumerate(self.data_generator.train_loader):
                batch = self.data_generator.create_batch(x0, device='cuda')
                batch.pop('x0')
                steps_printer.print_freq = 100
                progress_printer = lambda t_value: steps_printer(
                    batch_index, num_iterations, 
                    t_value, self.diffusion.time_distribution.num_steps,
                    'bash', process.rank, fprint=fprint
                )
                generated_objects = self.diffusion.sampling_procedure(
                    batch=batch, 
                    model=self.model,
                    progress_printer=progress_printer,
                    num_sampling_steps=self.config.num_sampling_steps
                )
                if batch_index == num_iterations - 1 and last_batch_size > 0:
                    generated_objects = generated_objects[:last_batch_size]
                self.data_generator.save_generated_objects(
                    generated_objects,
                    folder=self.folder,
                    rank=process.rank
                )
                if batch_index == num_iterations - 1:
                    break

        self.model.cpu()
        self.diffusion.cpu()
        pass


