#! /bin/bash


c2c hydra2code --input config/train_diffusion.yaml --output gbi_diff/utils/configs/train_diffusion.py
c2c hydra2code --input config/train_guidance.yaml --output gbi_diff/utils/configs/train_guidance.py
c2c hydra2code --input config/train_potential.yaml --output gbi_diff/utils/configs/train_potential.py

c2c file2code --input config/sampling_diffusion.yaml --output gbi_diff/utils/configs/sampling_diffusion.py    
c2c file2code --input config/sampling_mcmc.yaml --output gbi_diff/utils/configs/sampling_mcmc.py    