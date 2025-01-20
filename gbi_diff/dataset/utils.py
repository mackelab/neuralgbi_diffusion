import torch
from tqdm import tqdm


def generate_x_misspecified(x: torch.Tensor, diffusion_scale=0.5, max_steps=10000):
    """Generate misspecified x by diffusion from simulated x.

    Args:
        x (torch.Tensor): (n_observations, simulator_out_features)
        diffusion_scale (float, optional): _description_. Defaults to 0.5.
        max_steps (int, optional): _description_. Defaults to 10000.

    Returns:
        _type_: _description_
    """
    n_observations = x.shape[0]

    counter = 0
    x_min, x_max = x.min(0)[0], x.max(0)[0]
    x_std = x.std(0)

    mask = torch.zeros(n_observations)
    while (not (mask > 0).all()) and counter < max_steps:
        x += (
            ((torch.randn(x.shape) > 0).to(float) - 0.5)
            * 2
            * x_std
            * (1.0 - mask.unsqueeze(1))
            * diffusion_scale
        )
        mask = ((x < x_min).any(1) | (x > x_max).any(1)).to(float)
        counter += 1

    if counter - 1 == max_steps:
        msg = "Not all points out of bounds."
    else:
        msg = f"{counter} steps of diffusion for misspecified data."
    tqdm.write(msg)

    return x
