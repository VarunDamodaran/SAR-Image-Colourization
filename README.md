# SAR Image Colorization using Generative Adversarial Networks (GANs)

This repository contains the code and training pipeline for colorizing Synthetic Aperture Radar (SAR) images using a GAN-based architecture. The model employs a **U-Net generator** and a **PatchGAN discriminator**, trained on the **Sentinel-1** dataset to generate realistic colorized versions of grayscale SAR images.

## Overview

SAR images are inherently grayscale and often difficult to interpret visually. This project addresses that limitation by leveraging deep learning techniques to add color to SAR images, enhancing their usability in various applications such as remote sensing, environmental monitoring, and geospatial analysis.

We use a conditional GAN framework with:
- **Generator**: U-Net architecture to predict color information.
- **Discriminator**: PatchGAN to focus on high-frequency structures and encourage locally realistic outputs.

## Model Architecture

- **Generator**: U-Net
  - Takes a single-channel SAR (L) image as input.
  - Outputs predicted `ab` channels in Lab color space.
  - Skip connections help preserve spatial information.

- **Discriminator**: PatchGAN
  - Takes the concatenated (L, ab) image.
  - Outputs a matrix of realism scores for each patch.
  - Promotes local realism rather than global coherence alone.

- **Loss Functions**:
  - **L1 Loss** between real and generated `ab` channels.
  - **Adversarial Loss** from PatchGAN.
  - Optional: Total variation or perceptual loss (if used).

## Dataset

- **Sentinel-1** SAR images
- Images were preprocessed and converted to grayscale Lab space for the `L` channel.
- Ground truth color images were used to extract corresponding `ab` channels.

## Training Breakdown
This flowchart shows one step of the training cycle in the GAN framework:
<pre> 
┌────────────────────┐  
│  Sentinel-1 Images │  
└────────┬───────────┘  
         ▼  
┌────────────────────┐  
│  Convert to Lab    │  
│  (Extract L, ab)   │  
└────────┬───────────┘  
         ▼  
┌────────────────────┐  
│   Generator (U-Net)│  
│   Input: L         │  
│   Output: ab'      │  
└────────┬───────────┘  
         ▼  
┌────────────────────────────┐  
│   Concatenate L + ab'      │  
└────────┬───────────┬───────┘  
         │           │  
         │           ▼  
         │   ┌────────────────────┐  
         │   │  Discriminator     │  
         │   │  (PatchGAN)        │  
         │   │  Real vs Fake (L+ab│  
         │   └────────────────────┘  
         │           │  
         ▼           ▼  
┌────────────────────────────────────────┐  
│     Loss Calculation (L1 + GAN Loss)   │  
└────────────────────────────────────────┘  
         ▼  
┌────────────────────┐  
│   Backpropagation  │  
└────────┬───────────┘  
         ▼  
┌────────────────────┐  
│   Colorized Image  │  
│   L + ab' → RGB    │  
└────────────────────┘  
</pre>
This flowchart shows one step of the training cycle in the GAN framework:
<pre>
┌───────────────┐  
│  Input: L     │  
└──────┬────────┘  
       ▼  
┌───────────────┐        ┌───────────────┐  
│  Generator    │        │ Ground Truth  │  
│  (U-Net)      │        │ ab Channels   │  
└──────┬────────┘        └──────┬────────┘  
       │                        │  
       ▼                        ▼  
  Generated ab'          Real (L + ab)  
       │                        │  
       ▼                        ▼  
┌───────────────┐        ┌───────────────┐  
│  Fake Pair:   │        │  Real Pair:   │  
│  (L + ab')    │        │  (L + ab)     │  
└──────┬────────┘        └──────┬────────┘  
       ▼                        ▼  
       └─────►  Discriminator ◄─┘  
                  (PatchGAN)  
                       │  
                       ▼  
         ┌─────────────────────────┐  
         │  Compute GAN Loss       │  
         │  + L1 Loss (ab vs ab')  │  
         └──────────┬──────────────┘  
                    ▼  
           Backpropagation  
         (Update G and D params)  
</pre>

![A sample of the output](assets/output_traning.png)
## 🛠️ Tech Stack

- Python
- PyTorch
- NumPy
- OpenCV / PIL
- Matplotlib
