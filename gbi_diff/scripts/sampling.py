from typing import Dict
import torch
from gbi_diff.sampling.utils import create_potential_fn
from gbi_diff.utils.mcmc_config import Config
from pyro.infer import MCMC
from pyro import infer



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
        Dict[str, torch.Tensor]: dictionary with variable name and sampled variables. each sampled value has the shape: (n_samples, size, param_dim). The observed data was also added ot the result with shape (n_samples, observed_dim)
    """
    potential_func = create_potential_fn(checkpoint, config, x_o=None)
    n_samples = x_o.shape[0]
    
    samples = []
    for idx, x in enumerate(x_o):
        potential_func.update_x_o(x)

        kernel_cls = getattr(infer, config.kernel)
        kernel = kernel_cls(potential_fn=potential_func)
        mcmc = MCMC(
            kernel,
            num_samples=size,
            warmup_steps=config.warmup_steps,
            initial_params={"theta": potential_func.prior.sample()},
        )
        mcmc.run(x)
        
        samples.append(mcmc.get_samples())
        print(f"Completed: {idx+1}/{n_samples}")

    # assume all sample have the same keys
    res = {k: [] for k in samples[0].keys()}
    for k in res.keys():
        for sample in samples:
            res[k].append(sample[k])
        res[k] = torch.stack(res[k])
    # add observed data
    res["x_o"] = x_o

    return res