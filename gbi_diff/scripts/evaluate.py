from pathlib import Path
from typing import List

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


def evaluate_diffusion_sampling(
    diffusion_ckpt: str | Path,
    guidance_ckpt: str | Path,
    eval_config: DictConfig,
    output: str | Path = None,
    file_name: str = "eval_samples.pt",
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
        file_name (str, optional): _description_. Defaults to "eval_samples.pt".
    """
    eval_config: EvalDiffConfig = EvalDiffConfig.from_dict_config(
        eval_config, resolve=True
    )

    sampler = DiffusionSampler(
        diffusion_ckpt,
        guidance_ckpt,
        observed_data_file=eval_config.observed_data_file,
        beta=eval_config.betas[0],
    )
    print(yaml.dump(eval_config.to_container(), indent=4))
    
    cls_name = to_camel_case(eval_config.data_entity)
    cls_name = cls_name[0].upper() + cls_name[1:]
    dataset_cls = getattr(sbi_datasets, cls_name)
    dataset: sbi_datasets._SBIDataset = dataset_cls.from_file(
        eval_config.observed_data_file
    )

    param_samples = []
    obs_samples = []

    iterator = tqdm(eval_config.betas, desc=f"Beta: {eval_config.betas[0]}")
    for beta in iterator:
        iterator.set_description(f"Beta: {beta}")
        sampler.update_beta(beta)
        theta_pred = sampler.forward(eval_config.n_samples, quiet=1)
        x_pred = [
            dataset.sample_posterior(theta) for theta in theta_pred.permute((1, 0, 2))
        ]
        x_pred = torch.stack(x_pred).permute(1, 0, 2)
        param_samples.append(theta_pred)
        obs_samples.append(x_pred)
    param_samples = torch.stack(param_samples).detach()
    obs_samples = torch.stack(obs_samples).detach()
    
    res = {
        **eval_config.to_container(),
        "x_o": sampler.x_o,
        "theta": param_samples,
        "x_pred": obs_samples,
    }

    if output is None:
        output = sampler._get_default_path()
    elif isinstance(output, str):
        output = Path(output)
    output = output / file_name

    print(f"Save Eval samples at: {output}")
    save_torch(res, output)