# SAR Image Colorization

This repository contains a Synthetic Aperture Radar (SAR) image colorization project implemented using PyTorch. The model is based on a Generative Adversarial Network (GAN) with a U-Net architecture as the generator and a PatchGAN-based discriminator. The model is trained on the Sentinel-1 dataset of satellite images.

## Features
- Uses a GAN framework for SAR image colorization
- U-Net architecture as the generator
- PatchGAN-based discriminator
- Trained on Sentinel-1 satellite images
- Implements adversarial loss and L1 loss for training

## Dataset
The model is trained on the Sentinel-1 dataset. Ensure you have the dataset prepared in the appropriate format before training.

## Model Architecture
- **Generator:** U-Net architecture
- **Discriminator:** PatchGAN-based architecture

