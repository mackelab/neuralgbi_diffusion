import numpy as np
import torch


def dim_repeat(x: torch.Tensor, repeat: int, dim: int) -> torch.Tensor:
    """add a dimension in `x` at `dim` and repeat it `repeat` times

    Args:
        x (torch.Tensor):
        repeat (int): _description_
        dim (int): _description_

    Returns:
        torch.Tensor: _description_
    """
    x = torch.unsqueeze(x, dim=dim)

    x_repeat_dim = np.ones(len(x.shape), dtype=int)
    x_repeat_dim[dim] = repeat
    x = x.repeat(x_repeat_dim.tolist())
    return x
