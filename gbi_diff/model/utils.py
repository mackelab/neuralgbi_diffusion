import logging
from typing import Callable, Tuple
from torch import Tensor
from sbi.utils.sbiutils import handle_invalid_x
import torch

from gbi_diff.utils.metrics import compute_distances


def compute_standardizing_net_params(
    batch_t: Tensor,
    structured_dims: bool = False,
    min_std: float = 1e-7,
) -> Tuple[Tensor, Tensor]:
    """returns parameters to initialize the standardize network

    Args:
        batch_t: Batched tensor from which mean and std deviation (across
            first dimension) are computed.
        structured_dim: Whether data dimensions are structured (e.g., time-series,
            images), which requires computing mean and std per sample first before
            aggregating over samples for a single standardization mean and std for the
            batch, or independent (default), which z-scores dimensions independently.
        min_std:  Minimum value of the standard deviation to use when z-scoring to
            avoid division by zero.

    Returns:
        parameters for sbi.utils.sbiutils.Standardize network
    """

    is_valid_t, *_ = handle_invalid_x(batch_t, True)

    if structured_dims:
        # Structured data so compute a single mean over all dimensions
        # equivalent to taking mean over per-sample mean, i.e.,
        # `torch.mean(torch.mean(.., dim=1))`.
        t_mean = torch.mean(batch_t[is_valid_t])
    else:
        # Compute per-dimension (independent) mean.
        t_mean = torch.mean(batch_t[is_valid_t], dim=0)

    if len(batch_t > 1):
        if structured_dims:
            # Compute std per-sample first.
            sample_std = torch.std(batch_t[is_valid_t], dim=1)
            sample_std[sample_std < min_std] = min_std
            # Average over all samples for batch std.
            t_std = torch.mean(sample_std)
        else:
            t_std = torch.std(batch_t[is_valid_t], dim=0)
            t_std[t_std < min_std] = min_std
    else:
        t_std = 1
        logging.warning(
            """Using a one-dimensional batch will instantiate a Standardize transform
            with (mean, std) parameters which are not representative of the data. We
            allow this behavior because you might be loading a pre-trained. If this is
            not the case, please be sure to use a larger batch."""
        )

    return t_mean, t_std


def compute_multiplybymean_params(
    distance_func: Callable[[Tensor, Tensor], Tensor], x_target: Tensor, x_o: Tensor
) -> Tuple[Tensor, Tensor]:
    distances = compute_distances(distance_func, x_target, x_o)
    print(distances.shape)
    return distances.mean(), distances.std()
