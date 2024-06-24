**Training Script for SEMANTIFY**

This repository contains the end-to-end training script for our project, designed to run efficiently on Kaggle using a P100 GPU. The original model checkpoints were trained on a DGX server equipped with a single A100 GPU.

To facilitate a smoother introduction to our training pipeline, we have provided a Kaggle notebook. This notebook will help you quickly understand and get started with the training process.

*Disclaimer*
Despite using exact same random seed, slight performance mismatch may occur due to the usage of different GPUs (P100 vs. A100), different versions of CUDA, and PyTorch.

