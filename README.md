# Star-Shaped Denoising Diffusion Probabilistic Models
This repo contains the official PyTorch implementation for the paper [Star-Shaped Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2302.05259) -- approach to creating non-Gaussian diffusion models applicable to various non-euclidean manifolds.

by Andrey Okhotin\*, Dmitry Molchanov\*, Vladimir Arkhipkin, Grigory Bartosh, Viktor Ohanesian, Aibek Alanov, Dmitry Vetrov

assistent: Sergei Kholkin

<p align="center">
  <img src="https://github.com/andrey-okhotin/star-shaped/blob/main/results/demo.jpg" width="700" height="403">
</p>


## Abstract:
Denoising Diffusion Probabilistic Models (DDPMs) provide the foundation for
the recent breakthroughs in generative modeling. Their Markovian structure make it difficult to define DDPMs with distributions other than Gaussian or discrete. In this paper, we introduce Star-Shaped DDPM (SS-DDPM). Its star-shaped diffusion process allows us to bypass the need to define the transition probabilities or compute posteriors. We establish duality between star-shaped and specific Markovian diffusions for the exponential family of distributions, and derive efficient algorithms for training and sampling from SS-DDPMs. In the case of Gaussian distributions, SS-DDPM is equivalent to DDPM. However, SS-DDPMs provide a simple recipe for designing diffusion models with distributions such as Beta, von Mises—Fisher, Dirichlet, Wishart and others, which can be especially useful when data lies on a constrained manifold. We evaluate the model in different settings and find it competitive even on image data, where Beta SS-DDPM achieves results comparable to a Gaussian DDPM.


# Documentation

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)


## How to use this repo?
Main SS-DDPM logic described in "lib/diffusion" directory. This can be enough if you want to
* better understand the proposed approach
* incorporate SS-DDPM model to your local repository

Also you can find examples of using SS-DDPM on geodesic and synthetic data in directory "notebooks". If you want to reproduce our results, you can find examples of commands executions for experiments on CIFAR10 and Text8.

## Short content description
Repo structure:
* checkpoints - folder where model checkpoints will be saved during training pipelines
* datasets - data storage
* lib - folder with SS-DDPM implementation
* logs - where will be saved log info about running pipelines
* notebooks - folder for jupyter-notebook examples
* pretrained_models - storage of final model weights
* results - folder for saving different info about experiments
* saved_configs - storage for saving full experiments configs
* requirments.txt - dependency installation file


## Installing dependencies
This repo tested with torch==1.12.0+cu113 torchvision==0.13.0+cu113
```bash
git clone https://github.com/andrey-okhotin/star-shaped
cd star-shaped
pip install -r requirements.txt

# only if you don't have pytorch or your pytorch version < 1.11
pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113

# only for experiments with synthetic data, otherwise you can just comment all 'import npeet'
git clone https://github.com/gregversteeg/NPEET.git && cd NPEET && pip install . && cd ../ && rm -rf NPEET
```
Downloading content of **datasets** folder - necessary for all pipelines. This command may take about 5 minutes.
```bash
pip install py7zr gdown
rm -rf star-shaped/pretrained_models
gdown --fuzzy https://drive.google.com/file/d/1ndXOmbNXR6pwoJ5qs1gVP0eAKU_RAl6E/view?usp=sharing
py7zr x datasets.7z && rm datasets.7z && mv datasets star-shaped/datasets
```
Downloading content of **pretrained_models** folder - not necessary for training pipelines. This command may take about 3 minutes.
```bash
pip install py7zr gdown
rm -rf star-shaped/datasets
gdown --fuzzy https://drive.google.com/file/d/1Lebmsti31CwOFg4LYJYlWmlS7rGYQfVi/view?usp=sharing
py7zr x pretrained_models.7z && rm pretrained_models.7z && mv pretrained_models star-shaped/pretrained_models
```

## Available pipelines

#### Small experiments
Available for running from jupyter-notebook in directory SS_DDPM/notebooks. There you can find examples of training and sampling for
- Wishart SS-DDPM on synthetic data
- Dirichlet SS-DDPM on synthetic data
- DDPM on synthetic data
- Von Mises Fisher SS-DDPM on geodesic data

#### Large experiments
Available for running from bash in directory SS_DDPM
- training_cifar10   - train Beta SS-DDPM or DDPM with NCSN++ on CIFAR10 
- training_text8     - train Categorical SS-DDPM or D3PM with T5Encoder on Text8
- sampling_cifar10 - sample from Beta SS-DDPM or DDPM using defined pretrained model
- sampling_text8 - sample from Categorical SS-DDPM or D3PM using defined pretrained model
- estimating_nll_text8 - estimating Negative Log Likelihood in bits/dim of Categorical SS-DDPM or D3PM on defined part of Text8 dataset

Running command:
```bash
python lib/run_pipeline -gpu <gpu0_idx>_<gpu1_idx>_<gpu2_idx>  -pipeline <pipeline_name> -logs_file <name_of_txt_file_to_write_execution_info>  -port <available_port_for_processes_sync>   . . .   "other_pipeline_arguments"
```
Short usage example for running on 3 gpus on a single node:
```bash
python lib/run_pipeline -gpu 0_1_2  -pipeline train_cifar10 -logs_file logs_train_cifar10.txt -port 8890    . . . "other_pipeline_arguments"
```

##### Beta SS-DDPM on CIFAR10

Training Beta SS-DDPM on 4 Nvidia V100(need ~ 32Gb gpu memory). Checkpoints will be saved in the directory "checkpoints/train_beta_ss_cifar10". Loss graphics will be saved in the directory "results/train_beta_ss_cifar10".
```bash
python lib/run_pipeline.py -gpu 0_1_2_3 -port 8900 -pipeline training_cifar10 -diffusion beta_ss -loss KL_rescaled -save_folder train_beta_ss_cifar10 -logs_file logs_training_beta_ss_cifar10.txt
cp checkpoints/training_beta_ss_cifar10/NCSNpp_episode0_epoch1050_model.pt pretrained_models/ncsnpp-cifar10_beta-ss.pt
```
Sampling Beta SS-DDPM on 2 Nvidia V100. Results will be saved in the directory "results/sampling_beta_ss_cifar10/generated_samples".
```bash
python lib/run_pipeline.py -gpu 0_1 -port 8900 -pipeline sampling_cifar10 -diffusion beta_ss -num_sampling_steps 1000 -pretrained_model ncsnpp-cifar10_beta-ss.pt -num_samples 50000 -save_folder sampling_beta_ss_cifar10 -logs_file logs_sampling_beta_ss.txt
python -m pytorch_fid datasets/FID_cifar10_pack50000 results/sampling_beta_ss_cifar10/generated_samples
```
If you run exactly the same commands you will get FID ~ 3.24 .

##### Categorical SS-DDPM on Text8
Training Categorical SS-DDPM on 4 Nvidia A100 (need ~150Gb gpu memory). Checkpoints will be saved in the directory "checkpoints/training_categorical_ss_text8". Loss graphics will be saved in the directory "results/training_categorical_ss_text8".
```bash
python lib/run_pipeline.py -gpu 0_1_2_3 -port 8900 -pipeline training_text8 -diffusion categorical_ss -loss KL -save_folder training_categorical_ss_text8 -logs_file logs_training_categorical_ss.txt
cp checkpoints/training_categorical_ss_text8/T5Encoder_episode0_epoch2016_model.pt pretrained_models/t5base-text8_categorical-ss_fully-trained.pt
```
Estimating NLL in Categorical SS-DDPM on 3 Nvidia A100. Results will be saved in the directory "results/nll_estimations". 
```bash
python lib/run_pipeline.py -gpu 0_1_2 -port 8900 -pipeline estimating_nll_text8 -diffusion categorical_ss -pretrained_model t5base-text8_categorical-ss_fully-trained.pt -num_samples -1 -batch_size 1536 -dataset_part test -num_iwae_trajectories 1 -save_folder nll_text8_categorical-ss -logs_file logs_nll_text8_categorical_ss.txt
```
If you run exactly the same commands you will get NLL ~ 1.61 .




## Citation


```python
@inproceedings{okhotin2023star,
    author={Andrey Okhotin, Dmitry Molchanov, Vladimir Arkhipkin, Grigory Bartosh, Viktor Ohanesian, Aibek Alanov and Dmitry Vetrov},
    title={Star-Shaped Denoising Diffusion Probabilistic Models},
    booktitle={Advances in Neural Information Processing Systems},
    volume={36},
    year={2023}
}
```




