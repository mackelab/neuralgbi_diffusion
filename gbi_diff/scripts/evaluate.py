import json
from pathlib import Path
from typing import List

import h5py
import torch
from tqdm import tqdm
import yaml
from gbi_diff.sampling.utils import save_torch
from gbi_diff.scripts.sampling import diffusion_sampling
from omegaconf import DictConfig
from gbi_diff.utils.cast import to_camel_case
from gbi_diff.utils.evaluate_diffusion_config import Config as EvalDiffConfig
from gbi_diff.sampling.diffusion import DiffusionSampler
from gbi_diff.dataset import dataset as sbi_datasets
from gbi_diff.utils.train_diffusion_config import Config as DiffusionTrainConfig


def evaluate_diffusion_sampling(
    diffusion_ckpt: str | Path,
    guidance_ckpt: str | Path,
    eval_config: DictConfig,
    output: str | Path = None,
    file_name: str = "eval_samples.h5",
):
    """resulting torch save:

    {
    "betas": (n_betas, )
    "x_o": (n_x_o, observation_dim)
    "theta": (n_betas, n_samples, n_x_o, param_dim)
    "x_pred": (n_betas, n_samples, n_x_o, observation_dim)
    }

    Args:
        diffusion_ckpt (str | Path): _description_
        guidance_ckpt (str | Path): _description_
        eval_config (DictConfig): _description_
        output (str | Path, optional): _description_. Defaults to None.
        file_name (str, optional): _description_. Defaults to "eval_samples.h5".
    """
    eval_config: EvalDiffConfig = EvalDiffConfig.from_dict_config(
        eval_config, resolve=True
    )
    train_config: DiffusionTrainConfig = DiffusionTrainConfig.from_file(
        Path(diffusion_ckpt).parent.joinpath("config.yaml")
    )
    sampler = DiffusionSampler(
        diffusion_ckpt,
        guidance_ckpt,
        observed_data_file=eval_config.observed_data_file,
        gamma=eval_config.betas[0],
        normalize_data=train_config.dataset.normalize,
        extended_information=True
    )
    # sampler.x_o = sampler.x_o[:2]
    print(yaml.dump(eval_config.to_container(), indent=4))

    cls_name = to_camel_case(eval_config.data_entity)
    cls_name = cls_name[0].upper() + cls_name[1:]
    dataset_cls = getattr(sbi_datasets, cls_name)
    dataset: sbi_datasets._SBIDataset = dataset_cls.from_file(
        eval_config.observed_data_file
    )

    if output is None:
        output = sampler._get_default_path()
    elif isinstance(output, str):
        output = Path(output)
    output = output / file_name
    print(f"Save Eval samples at: {output}")
    output.parent.mkdir(exist_ok=True, parents=True)
    file = h5py.File(output, "w")
    file.attrs["x_o"] = sampler.x_o
    file.attrs["theta_o"] = sampler.theta_o
    for key, value in eval_config.to_container().items():
        file.attrs[key] = value
    theta_dim = sampler._guidance_model.hparams.theta_dim
    x_dim = sampler._guidance_model.hparams.simulator_out_dim
    n_x_o = len(sampler.x_o)
    param_samples = file.create_dataset(
        "theta_pred",
        (len(eval_config.betas), eval_config.n_samples, n_x_o, theta_dim),
        dtype="float32",
    )
    obs_gt = file.create_dataset(
        "x_gt",
        (n_x_o, eval_config.n_samples, x_dim),
        dtype="float32",
    )
    obs_samples = file.create_dataset(
        "x_pred",
        (len(eval_config.betas), eval_config.n_samples, n_x_o, x_dim),
        dtype="float32",
    )
    guidance_grads = file.create_dataset(
        "guidance_grads",
        (len(eval_config.betas), n_x_o, sampler._diff_model.diffusion_steps, eval_config.n_samples, theta_dim),
        dtype="float32",
    )
    diffusion_steps = file.create_dataset(
        "diffusion_steps",
        (len(eval_config.betas), n_x_o, sampler._diff_model.diffusion_steps, eval_config.n_samples, theta_dim),
        dtype="float32",
    )
    trajectory = file.create_dataset(
        "trajectory",
        (len(eval_config.betas), n_x_o, sampler._diff_model.diffusion_steps + 1, eval_config.n_samples, theta_dim),
        dtype="float32",
    )
    
    # add ground truth samples
    print("sample ground truth")
    for x_o_idx, theta in enumerate(sampler.theta_o):
        obs_gt[x_o_idx] = dataset.sample_posterior(theta[None].repeat(eval_config.n_samples, 1))

    print("Sample from predicted distribution")
    iterator = tqdm(eval_config.betas, desc=f"Beta: {eval_config.betas[0]}")
    for beta_idx, beta in enumerate(iterator):
        iterator.set_description(f"Beta: {beta}")
        sampler.update_gamma(beta)
        file = sampler.forward(
            eval_config.n_samples,
            quiet=1,
            h5_file=(file, slice(beta_idx + 1, beta_idx + 2)),
        )
        # guidance_grads[beta_idx] = sampler._info["guidance_grads"]
        # diffusion_steps[beta_idx] = sampler._info["diffusion_steps"]
        # trajectory[beta_idx] = sampler._info["trajectory"]
        
        for x_o_idx in range(n_x_o):
            obs_samples[beta_idx, :, x_o_idx] = dataset.sample_posterior(
                torch.from_numpy(param_samples[beta_idx, :, x_o_idx])
            )
    print("Finished saving")
    file.close()
