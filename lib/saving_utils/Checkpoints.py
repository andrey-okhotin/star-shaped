import torch
import json
import os
import shutil
from pathlib import Path
from ml_collections import ConfigDict

from models.Models import init_model
from saving_utils.get_repo_root import get_repo_root




class Checkpoints:
    
    def __init__(self, checkpoints_config, rank):
        """
        INPUT:
        
        <>  checkpoints_config = {
                'saving_freq' : (int) every 'saving_freq' epochs Chekpoints objects
                    would save model parameters,
                'folder' : (str) folder where model config and model parameters
                    will be saved
            }
        
        """
        self.rank = rank
        if self.rank != 0:
            return None
        
        period = checkpoints_config['saving_freq']
        self.saving_rule = lambda epoch: isinstance(epoch, int) and (epoch % period == 0)

        if os.path.isabs(checkpoints_config['folder']):
            checkpoints_config['folder'] = os.path.basename(os.path.normpath(checkpoints_config['folder']))
        checkpoints_path = os.path.join(get_repo_root(), '..', 'app')
        self.checkpoints_folder = os.path.join(checkpoints_path, checkpoints_config['folder'])
        
        if os.path.exists(self.checkpoints_folder):
            if 'reset_previous' in checkpoints_config.keys() and checkpoints_config['reset_previous']:
                shutil.rmtree(self.checkpoints_folder)
                os.mkdir(self.checkpoints_folder)
            else:
                print('Checkpoints folder already exists.')
                print('Existing checkpoints in folder would be replaced by their duplicates.')
        else:
            os.mkdir(self.checkpoints_folder)
        os.chmod(self.checkpoints_folder, 0o444)
        pass
        
    
    def create_checkpoint(self, model, optimizer, episode, epoch, last=False, tag='model'):
        if self.rank != 0:
            return None
        
        if self.saving_rule(epoch) or last:
            optimizer.switch_to_ema()
            model_name = model.model_name
            if last:
                model_name = model_name + f'_last_{tag}.pt'
            else:
                model_name = model_name + f'_episode{episode}_epoch{epoch}_{tag}.pt'
            torch.save(model.state_dict(), os.path.join(self.checkpoints_folder, model_name))
            config_name = model.model_name + f'_config.json'
            with open(os.path.join(self.checkpoints_folder, config_name), 'w') as json_file:
                json.dump(dict(model.model_config.items()), json_file)
            optimizer.switch_from_ema()
        pass

    
    def load_checkpoint(self, episode, epoch, last=False, tag='model'):
        files_list = os.listdir(self.checkpoints_folder)
        for file_name in files_list:
            if file_name[-12:] == '_config.json':
                config_name = file_name
                model_name = file_name[:-12]
                break
        
        with open(os.path.join(self.checkpoints_folder, config_name)) as json_file:
            model_config = json.load(json_file)
        config = {
            'model_name' : model_name,
            'model_config' : ConfigDict(model_config)
        }
        model = init_model(config)
        
        if last:
            model_file = model_name + f'_last_{tag}.pt'
        else:
            model_file = model_name + f'_episode{episode}_epoch{epoch}_{tag}.pt'
        model_file = os.path.join(self.checkpoints_folder, model_file)
        model.load_state_dict(torch.load(model_file))
        return config, model
    
    
    def extract_episodes_and_epochs(self):
        # collect
        episodes, epochs = [], []
        for checkpoint_file in os.listdir(self.checkpoints_folder):
            if checkpoint_file.split('.')[-1] == 'pt':
                episode, epoch = checkpoint_file.split('_')[1:3]
                if episode != 'last':
                    episodes += [ int(episode.replace('episode', '')) ]
                    epochs += [ int(epoch.replace('epoch', '')) ]
        # sort            
        indexes = torch.tensor([episodes, epochs])
        indexes = indexes[:,torch.argsort(indexes[0])]
        episodes, epochs = indexes
        for episode in episodes.unique():
            mask = (episode == episodes)
            epochs[mask] = epochs[mask].sort()[0]
            
        # pack
        episodes, epochs = episodes.tolist(), epochs.tolist()
        result_indexes = []
        for episode, epoch in zip(episodes, epochs):
            result_indexes += [ (episode, epoch) ]
            
        return result_indexes
    
    
    def clean_checkpoints_folder(self, tag='model'):
        removed = 0
        for f_name in os.listdir(self.checkpoints_folder):
            cur_tag = f_name.split('.')[0].split('_')[-1]
            if cur_tag == tag:
                os.remove(os.path.join(self.checkpoints_folder, f_name))
                removed += 1
        print(f'successfully remove {removed} checkpoints')
        pass
        
        
        
