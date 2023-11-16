# Star-Shaped Denoising Diffusion Probabilistic Models
This repo contains the official PyTorch implementation for the paper [Star-Shaped Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2302.05259)

by Andrey Okhotin*, Dmitry Molchanov*, Vladimir Arkhipkin, Grigory Bartosh, Viktor Ohanesian, Aibek
Alanov, Dmitry Vetrov

## Abstract:
Denoising Diffusion Probabilistic Models (DDPMs) provide the foundation for
the recent breakthroughs in generative modeling. Their Markovian structure make
it difficult to define DDPMs with distributions other than Gaussian or discrete.
In this paper, we introduce Star-Shaped DDPM (SS-DDPM). Its star-shaped
diffusion process allows us to bypass the need to define the transition probabilities
or compute posteriors. We establish duality between star-shaped and specific
Markovian diffusions for the exponential family of distributions, and derive efficient
algorithms for training and sampling from SS-DDPMs. In the case of Gaussian
distributions, SS-DDPM is equivalent to DDPM. However, SS-DDPMs provide a
simple recipe for designing diffusion models with distributions such as Beta, von
Mises—Fisher, Dirichlet, Wishart and others, which can be especially useful when
data lies on a constrained manifold. We evaluate the model in different settings
and find it competitive even on image data, where Beta SS-DDPM achieves results
comparable to a Gaussian DDPM.


# Release Date: Nov 24 '23
Do not miss!)
