# Neural GBI on Diffusion Processes

## Overview

This repository implements an advanced pipeline for posterior sampling in simulation-based inference settings using diffusion models. The project combines neural guidance and denoising models to enable robust Bayesian inference for scientific simulators. Additionally, it includes a reference implementation of the Generalized Bayesian Inference (GBI) pipeline from [Generalized Bayesian Inference for Scientific Simulators via Amortized Cost Estimation](https://arxiv.org/abs/2305.15208) (Gao et al., 2023) for benchmarking purposes.

## Key Features

- Diffusion-based posterior sampling pipeline
- Neural guidance and denoising model training
- Implementation of multiple scientific simulators
- MCMC sampling with NUTS kernel
- Comprehensive benchmarking tools
- Support for custom datasets

## Mathematical Framework

### Diffusion Process

The pipeline implements conditional guidance-controlled diffusion sampling based on [Score-Based Generative Modeling through Stochastic Differential Equations](https://arxiv.org/abs/2011.13456) (Song et al., 2020). The posterior gradient is decomposed as:

$$
\nabla_{\theta_\tau} \log(p_\psi(\theta_\tau | x)) =  \nabla_{\theta_\tau} \log(p_\psi(x_t | \theta_\tau )) + \nabla_{\theta_\tau} \log(p_\psi(\theta_\tau))
$$

where:

- $p_\psi(x_t | \theta_\tau ) = \frac{1}{Z} \exp(- \beta s_\psi(\theta_\tau, x_t, \tau))$
- $\nabla_{\theta_\tau} \log(p_\psi(\theta_\tau)) = f_\psi(\theta_\tau, \tau)$

## Installation

### Prerequisites

- Python >= 3.12
- pip
- git

### Setup Steps

1. Clone the repository:

```bash
git clone git@github.com:mackelab/neuralgbi_diffusion.git
```

2. Install Poetry (dependency management):

```bash
pip install poetry
```

3. Install dependencies:

```bash
poetry install --no-root
```

## Usage Guide

### Command Line Interface

The project provides a unified CLI interface for all operations:

```bash
python -m gbi_diff <action> <options>
```

Use `--help` or `-h` with any command to see available options.

### Data Generation

#### Available Simulators

The pipeline supports multiple scientific simulators:

- Two Moons (`two_moons`)
- SIR Epidemiological Model (`SIR`)
- Lotka-Volterra Population Dynamics (`lotka_volterra`)
- Inverse Kinematics (`inverse_kinematics`)
- Gaussian Mixture (`gaussian_mixture`)
- Linear Gaussian (`linear_gaussian`)
- Uniform 1D (`uniform`)

#### Generate Datasets

Generate data for a specific simulator:

```bash
python -m gbi_diff generate-data --dataset-type <type> --size <n_samples> --path data/
```

Recommended dataset sizes:

- Training: 10,000 samples
- Validation: 1,000 samples
- Observed data: 10 samples

For bulk dataset generation, use the provided script:

```bash
./generate_datasets.sh
```

### Custom Dataset Format

When adding custom datasets (*.pt files), include the following fields:

| Field                | Description                    | Shape/Type                      |
| -------------------- | ------------------------------ | ------------------------------- |
| _theta               | Parameter features             | (n_samples, n_param_features)   |
| _x                   | Simulator outcomes             | (n_samples, n_sim_out_features) |
| _target_noise_std    | Noise standard deviation       | float                           |
| _seed                | Random seed                    | int                             |
| _diffusion_scale     | Misspecification parameter     | float                           |
| _max_diffusion_steps | Misspecification parameter     | int                             |
| _n_misspecified      | Number of misspecified samples | int                             |
| _n_noised            | Number of noised samples       | int                             |

## Model Training

### Training the Guidance Model

The guidance model ($s_\psi(\theta_\tau, x_t, \tau)$) is trained using a modified loss function:

$$
\mathcal{L} = \mathbb{E}_{\theta, x \sim \mathcal{D}, \tau\sim U_{[0, T - 1]}}[||s_\psi(\theta_\tau, x_t, \tau) - d(x, x_t)||^2]
$$

Train the guidance model:

```bash
python -m gbi_diff train-guidance
```

Configuration: Modify `config/train_guidance.yaml`

### Training the Diffusion Model

The diffusion model ($f_\psi(\theta_\tau, \tau)$) follows the approach from [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) (Ho et al., 2020):

$$
\begin{aligned}
\mathcal{L} &= \mathbb{E}[|| \epsilon - f_\psi(\theta_\tau, \tau)||^2] \\
\theta_\tau &= \sqrt{\bar{\alpha_\tau}} \theta_0 + \sqrt{1 - \bar{\alpha_\tau}} \tau
\end{aligned}
$$

Train the diffusion model:

```bash
python -m gbi_diff train-diffusion
```

Configuration: Modify `config/train_diffusion.yaml`

## Sampling Methods

### Diffusion Sampling

Sample from the posterior distribution:

```bash
python -m gbi_diff diffusion-sample --diffusion-ckpt <path> --guidance-ckpt <path> --n-samples <count> [--plot]
```

Configuration: Modify `config/sampling_diffusion.yaml`

Key parameter: `beta` controls sample diversity

### GBI with MCMC Sampling

Train the potential function:

```bash
python -m gbi_diff train-potential
```

Sample using MCMC with NUTS kernel:

```bash
python -m gbi_diff mcmc-sample --checkpoint <path> --size <count> [--plot]
```

## Project Structure

```bash
├── config/                     # Configuration files
│   ├── train_guidance.yaml
│   ├── train_diffusion.yaml
│   ├── sampling_diffusion.yaml
│   └── train.yaml
├── data/                       # Dataset storage
├── gbi_diff/                   # Core implementation
│   ├── dataset                 # Dataset handling and implementation
│   ├── diffusion               # Toolset for training guidance and denoiser
│   ├── model                   # Network architectures + lighting module
│   ├── sampling                # Toolset for sampling
│   ├── scripts                 # Functions called by the entrypoint
│   └── utils                   # Utility
│   ├── __init__.py     
│   ├── __main__.py             # Main entrypoint (generated by pyargwriter)
│   ├── entrypoint.py           # Contains entrypoint class
├── results/                    # Training and sampling outputs
├── generate_datasets.sh        # Dataset generation script
├── poetry.lock                 # Dependency lock file
└── pyproject.toml              # Project metadata and dependencies
```

## Contributing

Please ensure any contributions:

1. Follow the existing code style
2. Include appropriate tests
3. Update documentation as needed
4. Maintain compatibility with the existing data format

## License

MIT License

## Citations

If you use this code in your research, please cite:

```bibtex
@misc{vetter2025gbidiff,
  title={Generalized Diffusion Simulation Based Inference},
  author={Vetter, Julius and Uhrich, Robin},
  year={2025},
  url={https://github.com/mackelab/neuralgbi_diffusion}
}
```
