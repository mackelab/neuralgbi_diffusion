import os
from typing import Dict
import torch

# from gbi_diff.sampling import PotentialFunc
from gbi_diff.utils.sampling_mcmc_config import Config
from gbi_diff.sampling import prior_distr


def load_observed_data(path: str) -> torch.Tensor:
    """loads a torch file with observed data inside. Please make sure the

    Args:
        path (str): path to observed data file

    Returns:
        torch.Tensor: tensor with observed data (n_samples, n_features)
    """
    content = torch.load(path, map_location="cpu", weights_only=True)
    try:
        x_o = content["_x"]
    except KeyError:
        raise ValueError(
            f"The given file: `{path}` has to contain the key `_x` for observed data"
        )

    return x_o


def get_sample_path(
    checkpoint: str,
    file_name: str,
    output: str = None,
):
    if output is None:
        directory = ".".join(checkpoint.split(".")[:-1])
        split = directory.split("/")
        split[-1] = "samples_" + split[-1]
        output = "/".join(split) + "/" + file_name
    else:
        output = output.rstrip("/") + "/" + file_name
    return output


def save_samples(
    samples: Dict[str, torch.Tensor],
    checkpoint: str,
    output: str = None,
    file_name: str = "samples.pt",
):
    """save samples as pt file.

    Args:
        samples (Dict[str, torch.Tensor]): sampled data
        checkpoint (str): path to checkpoint
        output (str, optional): output directory. If none is given it will the checkpoint directory. Defaults to None.
        file_name (str, optional): How to name the file. Has to end with ".pt"
    """
    output = get_sample_path(str(checkpoint), file_name, output)
    directory = "/".join(output.split("/")[:-1])
    if not os.path.exists(directory):
        os.makedirs(directory)
    print(f"Save samples at: {output}")
    torch.save(samples, output)
