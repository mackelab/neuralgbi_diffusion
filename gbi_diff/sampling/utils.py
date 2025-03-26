from datetime import datetime
import os
from pathlib import Path
from typing import Dict, Tuple
import torch
from torch import Tensor


def load_observed_data(path: str) -> Tuple[Tensor, Tensor]:
    """loads a torch file with observed data inside. Please make sure the

    Args:
        path (str): path to observed data file

    Returns:
        Tuple[Tensor, Tensor]:
            - tensor with observed data (n_samples, n_features)
            - tensor with measured theta (n_samples, n_params)
    """
    content = torch.load(path, map_location="cpu", weights_only=True)
    try:
        x_o = content["_x"]
        theta = content["_theta"]
    except KeyError:
        raise ValueError(
            f"The given file: `{path}` has to contain the key `_x` for observed data"
        )

    return x_o, theta


def get_sample_path(
    checkpoint: Path,
    output: str = None,
) -> Path:
    if output is None:
        directory = checkpoint.parent
        file_name = checkpoint.stem
        sample_dir = "samples_" + file_name
        output = directory / sample_dir
    else:
        output = Path(output.rstrip("/") + "/")
    return output


def save_torch(
    samples: Dict[str, torch.Tensor],
    output: Path,
):
    """save samples as pt file.

    Args:
        samples (Dict[str, torch.Tensor]): sampled data
        checkpoint (str): path to checkpoint
        output (Path, optional): output directory. If none is given it will the checkpoint directory. Defaults to None.
        file_name (str, optional): How to name the file. Has to end with ".pt"
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(samples, output)


def get_datetime_str() -> str:
    """returns datetime as string

    Returns:
        str: %Y_%m_%d__%H_%M_%S
    """
    today = datetime.now()
    s = today.strftime("%Y_%m_%d__%H_%M_%S")
    return s
