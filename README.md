# Neural GBI on Diffusion Processes

This repository contains a pipeline to denoiser and guidance models for generalized simulation based inference as well as tooling for sampling  from the resulting posterior distribution.

Additionaly this repository contains a small implementation of the GBI pipeline presented in [Generalized Bayesian Inference for Scientific Simulators via Amortized Cost Estimation](https://arxiv.org/abs/2305.15208) by Richard Gao, Michael Deistler and Jakob H. Macke to benchmark our results.

## Key Features

- Diffusion-based posterior sampling pipeline
- Neural guidance and denoising model training
- Implementation of multiple scientific simulators
- Support for custom datasets

## Installation

This repository is self contained and does not need additional submodules and can be download / cloned form GitHub:

```bash
git clone git@github.com:mackelab/neuralgbi_diffusion.git
```

After downloading and setting up a base version of you favorite python environment (required python >3.12) we use poetry to manage and install all python packages.

```bash
pip install poetry
poetry install --no-root
```

After this step you are good to go to interact with the environment.

## Usage

To interact with the pipeline we bundle all actions in one entrypoint:

```bash
python -m gbi_diff <action> <options>
```

If you would like to know more about the available actions and options you can always use `--help` or `-h`

## Datasets

To use the repository you have to generate the data you would like to use. To generate samples you can use the CLI of the pipeline by calling

```bash
python -m gbi_diff generate-data --dataset-type <type> --size <n_samples> --path data/
```

We to test our results we use data from following simulators with the corresponding `dataset-type`:

- Two Moons: `two_moons`
- SIR: `SIR`
- Lotka Volterra: `lotka_volterra`
- Inverse Kinematics: `inverse_kinematics`
- Gaussian Mixture: `gaussian_mixture`
- Linear Gaussian: `linear_gaussian`
- Uniform 1D: `uniform`

usually we use `10.000` samples for training, `1000` samples for validation. For the observed data we usually use `10` samples.
If you would like to create all datasets at once use the `generate_dataset.sh` bash script.

If you would like to add you own dataset in the form of `*.pt` files please make sure they have the following key and value pairs:

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

## Diffusion

The main contribution of this repository is a pipeline for training and sampling a posterior distribution in a simulation based inference setting.
For this we have to train two models:

1. A diffusion time dependent guidance: $s_\psi(\theta_\tau, x_t, \tau)$ with $x_t$ as the target $x$
2. A denoising prior: $f_\psi(\theta_\tau, \tau)$

As laid out in [Song et al. 2020](https://arxiv.org/abs/2011.13456) you could do conditioned guidance controlled diffusion sampling as:

$$
\begin{aligned}
\nabla_{\theta_\tau} \log(p_\psi(\theta_\tau | x)) &=  \nabla_{\theta_\tau} \log(p_\psi(x_t | \theta_\tau )) + \nabla_{\theta_\tau} \log(p_\psi(\theta_\tau)) \\
\nabla_{\theta_\tau} \log(p_\psi(\theta_\tau)) &= f_\psi(\theta_\tau, \tau) \\
p_\psi(x_t | \theta_\tau ) &= \frac{1}{Z} \exp(- \beta s_\psi(\theta_\tau, x_t, \tau)) \\
\nabla_{\theta_\tau} \log(p_\psi(x_t | \theta_\tau )) &= - \beta \nabla_{\theta_\tau} s_\psi(\theta_\tau, x_t, \tau)
\end{aligned}
$$

### Train Guidance

We train the guidance $s_\psi(\theta_\tau, x_t, \tau)$ in the same fashion as [Gao et al. 2023](https://arxiv.org/abs/2305.15208) but now $\theta$ noised within a diffusion schedule as described from [Ho et al. 2020](https://arxiv.org/abs/2006.11239).
Therefor the loss guidance loss function is only slightly modified:

$$
\mathcal{L} = \mathbb{E}_{\theta, x \sim \mathcal{D}, \tau\sim U_{[0, T - 1]}}[||s_\psi(\theta_\tau, x_t, \tau) - d(x, x_t)||^2]
$$

To train the guidance model you should call:

```bash
python -m gbi_diff train-guidance
```

Important to note is hereby the [config file](config/train_guidance.yaml) you are passing into the function. Depending on which dataset you would like to train you have to adapt either the `data_entity` parameter or just adapt the train and test file name in the config directly.

### Train Diffusion

We train the diffusion model $f_\psi(\theta_\tau, \tau)$ in the same fashion as [Ho et al. 2020](https://arxiv.org/abs/2006.11239). And reuse the diffusion loss function:

$$
\begin{aligned}
\mathcal{L} &= \mathbb{E}[|| \epsilon - f_\	(\theta_\tau, \tau)||^2] \\
\theta_\tau &= \sqrt{\bar{\alpha_\tau}} \theta_0 + \sqrt{1 - \bar{\alpha_\tau}} \tau
\end{aligned}
$$

To train the guidance model you should call:

```bash
python -m gbi_diff train-diffusion
```

Important to note is hereby the [config file](config/train_diffusion.yaml) you are passing into the function. Depending on which dataset you would like to train you have to adapt either the `data_entity` parameter or just adapt the train and test file name in the config directly.

### Sample from Diffusion

To finally sample from the posterior distribution you have to point towards the checkpoints of the guidance and diffusion model you would like to use. Please be aware, that diffusion, guidance and the dataset you would like to sample for have to be from the same data entity (for example: `tow_moons`).
To start the sampling process please call:

```bash
python -m gbi_diff diffusion-sample --diffusion-ckpt <guidance-ckpt> --guidance-ckpt <guidance-ckpt> --n-samples <n-samples> (--plot)
```

If you would like to adapt the config you can modify the [sampling_diffusion.yaml](config/sampling_diffusion.yaml). An important parameter hereby is `beta`. Beta controls the variety of samples you will get out of the sampling pipeline.

## GBI

To compare our results we implemented the generalized simulation based inference pipeline with sampling as presented in [Gao et al. 2023](https://arxiv.org/abs/2305.15208).

### Train Potential Function

To have a guidance function you can plug into the MCMC sampler you first have to train one. In order to do so please call:

```bash
python -m gbi_diff train-potential
```

If you would like to adapt parameters for the potential function please adapt them in the corresponding [config file](config/train.yaml). In  order to be comparable to the diffusion process we left all parameters as similar as possible to the [guidance config file](config/train_guidance.yaml).

### MCMC Sample

If you would like to sample from the trained potential function with MCMC sampling and a NUTS kernel you can call the pipeline with:

```bash
python -m gbi_diff mcmc-sample --checkpoint <potential-function-ckpt>  --size <n-samples> (--plot)
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
@misc{uhrich2025gbidiff,
  title={Generalized Diffusion Simulation Based Inference},
  author={Uhrich, Robin and Vetter, Julian},
  year={2025},
  url={https://github.com/mackelab/neuralgbi_diffusion}
}
```
