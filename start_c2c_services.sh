#! /bin/bash

config2class stop-all
config2class start-service --input config/train.yaml --output gbi_diff/utils/train_config.py
config2class start-service --input config/train_guidance.yaml --output gbi_diff/utils/train_guidance_config.py
config2class start-service --input config/train_diffusion.yaml --output gbi_diff/utils/train_diffusion_config.py
config2class start-service --input config/sampling_mcmc.yaml --output gbi_diff/utils/sampling_mcmc_config.py
config2class start-service --input config/sampling_diffusion.yaml --output gbi_diff/utils/sampling_diffusion_config.py