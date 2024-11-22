from typing import Dict
import torch
from gbi_diff.sampling import PotentialFunc
from gbi_diff.utils.mcmc_config import Config
from gbi_diff.sampling import prior_distr
from config2class.utils import deconstruct_config
from pyro.infer import MCMC
from pyro import infer


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


def save_samples(samples: Dict[str, torch.Tensor], checkpoint: str, output: str = None):
    """save samples as pt file.

    Args:
        samples (Dict[str, torch.Tensor]): sampled data
        checkpoint (str): path to checkpoint
        output (str, optional): output directory. If none is given it will the checkpoint directory. Defaults to None.
    """
    if output is None:
        directory = ".".join(checkpoint.split(".")[:-1])
        split = directory.split("/")
        split[-1] = "samples_" + split[-1]
        output = "/".join(split) + "/samples.pt"
    else:
        output = output.rstrip("/") + "/samples.pt"
    torch.save(samples, output)



def sample_posterior(
    checkpoint: str, x_o: torch.Tensor, config: Config, size: int = 100
) -> Dict[str, torch.Tensor]:
    """sample form posterior

    Args:
        checkpoint (str): path to model checkpoint
        x_o (torch.Tensor): observed data (n_samples, n_features)
        config (Config): mcmc config
        size (int, optional): how many sample you would like to samples. Defaults to 100.

    Returns:
        Dict[str, torch.Tensor]: dictionary with variable name and sampled variables. each value has the shape: (n_samples, size, param_dim)
    """
    prior_config = getattr(config, config.prior)
    prior_cls = getattr(prior_distr, config.prior)
    prior = prior_cls(**deconstruct_config(prior_config))

    samples = []
    for x in x_o:
        potential_func = PotentialFunc(
            checkpoint=checkpoint, prior=prior, x_o=x, beta=config.beta
        )

        kernel_cls = getattr(infer, config.kernel)
        kernel = kernel_cls(potential_fn=potential_func)
        mcmc = MCMC(
            kernel,
            num_samples=size,
            warmup_steps=config.warmup_steps,
            initial_params={"theta": prior.sample()},
        )
        mcmc.run(x)
        
        samples.append(mcmc.get_samples())

    # assume all sample have the same keys
    res = {k: [] for k in samples[0].keys()}
    for k in res.keys():
        for sample in samples:
            res[k].append(sample[k])
        res[k] = torch.stack(res[k])

    return res