import torch

from metrics_and_losses.MetricsClasses import (
    KL,
    KL_rescaled,
    NLL
)




class MetricsTracker:

    def __init__(self, metrics_config):
        """
        INPUT:
        
        <>  metrics_config = {
                'validation_freq' : (int) - frequency of validation stage,
                'metrics_list' = [ {'method', ...}, {}, ... ] -
                    list of different metrics will be computed
                    at 'train' and/or 'validation', and/or 'train_validation' stages
                    {
                        'method' : (str) - metric name,
                        'mode' : (list) - stages on which metric will be computed
                            may be ('train', 'validation', 'train_validation')s
                        'episode' : (list) - episodes on which metric will be computed
                        'freq' : (int)
                    }
            }
        """
        self.metrics_config = metrics_config
        switch_metric = {
            'KL'                 : KL,
            'KL_rescaled'        : KL_rescaled,
            'NLL'                : NLL
        }
        self.metrics_tracker = {}
        for metric in self.metrics_config['metrics_list']:
            self.metrics_tracker[metric['method']] = {
                'metric' : switch_metric[metric['method']](),
                'freq_func' : lambda epoch: ((epoch+1) % metric['freq'] == 0),
                'episodes' : metric['episodes'],
                'modes' : metric['mode']
            }


    def compute_metrics(
        self, 
        batch,
        diffusion,
        train_object,
        mode, 
        episode, 
        epoch
    ):
        with torch.no_grad():
            for metric_card in self.metrics_tracker.values():
                if (mode in metric_card['modes'] and episode in metric_card['episodes']
                    and metric_card['freq_func'](epoch)):
                    metric_card['metric'].batch_metric_update(
                        batch, 
                        diffusion, 
                        train_object, 
                        mode
                    )
        pass


    def get_accumulated_metrics(self):
        metrics = {}
        for metric_method, metric_card in self.metrics_tracker.items():
            metric_val = metric_card['metric'].get_metric()
            if not (metric_val is None):
                metrics[metric_method] = metric_val
        return metrics


    def validation_rule(self, epoch):
        freq = self.metrics_config['validation_freq']
        return (epoch+1) % freq == 0

