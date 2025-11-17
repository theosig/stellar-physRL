# stellar-physRL
Code acompaning the manuscript "Toward model-free stellar chemical abundances", aa55376-25

This repository contains the implementation of the physics-constrained, self-supervised representation learning framework introduced in the paper. The method combines variational autoencoders with physically motivated inductive biases to learn interpretable, element-specific latent dimensions directly from stellar spectra, enabling:

Disentangled chemical representations

Noise-robust latent features aligned with abundance variations

Physically meaningful anomaly detection

A step toward model-free chemical abundance inference

The paper introduces and validates the framework on synthetic spectra and demonstrates applications to identifying chemically anomalous stars (e.g., α-poor, metal-poor, and CEMP stars).

If you use this code, please cite the paper:

“Toward model-free stellar chemical abundances”
A&A, accepted (aa55376-25).
Preprint: https://arxiv.org/abs/2511.09733
