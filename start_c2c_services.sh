#! /bin/bash

config2class start-service --input config/train.yaml --output gbi_diff/utils/train_config.py
config2class start-service --input config/train_theta_noise.yaml --output gbi_diff/utils/train_theta_noise_config.py
config2class start-service --input config/mcmc.yaml --output gbi_diff/utils/mcmc_config.py