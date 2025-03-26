from typing import Callable
from torch import Tensor
import torch
from torch import Tensor
import torch
from tqdm import tqdm


def batch_correlation(x1: Tensor, x2: Tensor) -> Tensor:
    """compute correlation of two batched tensors x1 and x2

    Args:
        x1 (Tensor): (batch_size, feature_dim)
        x2 (Tensor): (batch_size, feature_dim)

    Returns:
        Tensor: (batch_size, )
    """
    x = torch.stack([x1, x2])

    x = x - x.mean(axis=-1, keepdim=True)
    cov = torch.bmm(x.permute(1, 0, 2), x.permute(1, 2, 0))

    denominator = cov[:, 0, 1]
    numerator = torch.sqrt(cov[:, 0, 0] * cov[:, 1, 1])
    corr = denominator / numerator
    return corr


## MSE
def mse_dist(x_target: Tensor, x_o: Tensor) -> Tensor:
    """
    Computes the Mean Squared Error (MSE) distance between samples.

    Args:
        x_target (Tensor): A tensor of shape (n_target, x_dim).
        x_o (Tensor): A tensor of shape (x_dim) or (n_target, x_dim).

    Returns:
        Tensor: MSE values of shape [num_thetas].
    """
    mse = torch.square(x_target - x_o).mean(dim=-1)  # Average over data dimensions.
    return mse # .mean(dim=-1, keepdim=True)  # Monte-Carlo average


## MAE
def mae_dist(xs: Tensor, x_o: Tensor) -> Tensor:
    """
    Computes the Mean Absolute Error (MAE) distance between samples.

    Args:
        xs (Tensor): A tensor of shape [num_thetas, num_xs, num_x_dims].
        x_o (Tensor): A tensor of shape [num_xs, num_x_dims].

    Returns:
        Tensor: MAE values of shape [num_thetas].
    """
    mae = torch.abs(xs - x_o).mean(dim=2)  # Average over data dimensions.
    return mae.mean(dim=1)  # Monte-Carlo average


## MMD
def _sample_based_mmd(x: Tensor, y: Tensor, scale: float = 0.01) -> Tensor:
    """
    Computes the Maximum Mean Discrepancy (MMD) between two samples.

    Args:
        x (Tensor): First sample of shape [batch_dim, num_xs, num_x_dims].
        y (Tensor): Second sample of shape [batch_dim, num_xs, num_x_dims].
        scale (float): Scaling factor for the RBF kernel. Default is 0.01.

    Returns:
        Tensor: MMD value.
    """
    term1 = _sample_based_mmd_marginal(x, x, scale=scale)
    term2 = _sample_based_mmd_marginal(x, y, scale=scale)
    term3 = _sample_based_mmd_marginal(y, y, scale=scale)
    return term1 + term3 - 2 * term2


def _sample_based_mmd_marginal(x: Tensor, y: Tensor, scale: float = 0.01) -> Tensor:
    """
    Computes marginal MMD by summing over each dimension.

    Args:
        x (Tensor): First sample of shape [batch_dim, num_xs, num_x_dims].
        y (Tensor): Second sample of shape [batch_dim, num_xs, num_x_dims].
        scale (float): Scaling factor for the RBF kernel. Default is 0.01.

    Returns:
        Tensor: Marginal MMD value.
    """
    num_x = x.shape[1]
    num_y = y.shape[1]
    xo1 = x.repeat((1, num_y, 1))
    xo2 = y.repeat_interleave((num_x), dim=1)
    distances = torch.exp(-scale * torch.square(xo1 - xo2))
    average_dist = distances.prod(dim=2).mean(dim=1)
    return average_dist


def pairwise_mmd_dist(x_target: Tensor, x_o: Tensor) -> Tensor:
    assert x_target.shape == x_o.shape, "xs and x_o must have identical shapes"
    mmds = _sample_based_mmd(x_target, x_o)
    return mmds


def mmd_dist(x_target: Tensor, x_o: Tensor) -> Tensor:
    """
    Computes the Maximum Mean Discrepancy (MMD) distance between samples.

    Args:
        x_target (Tensor): A tensor of shape (batch_dim, n_trials, x_dim).
        x_o (Tensor): A tensor of shape (n_trials, x_dim) or (1, n_trials, x_dim).

    Returns:
        Tensor: MMD values of shape (batch_dim,).
    """
    assert len(x_o.shape) > 1, "x_o must be at least 2D. (n_trials, x_dim)"
    # assert len(xs.shape) == 4, "xs must have shape [num_thetas, 1, num_xs, num_x_dims]."
    assert x_target.shape[1] > 1, "More than one trial in x_target is required."

    # Compute all pairwise MMDs between xs and x_o.
    if len(x_o.shape) > 2:              
        x_o = x_o.squeeze()
    assert x_o.shape[0] > 1, "x_o must have more than 1 data point."
    # mmds = torch.stack([_sample_based_mmd(x, x_o) for x in x_target])
    x_o = x_o[None].repeat((len(x_target), 1, 1))
    mmds = pairwise_mmd_dist(x_target, x_o)
    return mmds


def compute_distances(
    distance_func: Callable[[Tensor, Tensor], Tensor],
    x_target: Tensor,
    x_o: Tensor,
    memory_efficient: bool = True,
    quiet: bool = False,
) -> Tensor:
    """_summary_

    Args:
        distance_func (Callable[[Tensor, Tensor], Tensor]): distance function
        x_target (Tensor): (n_target, n_trials, x_dim)
        x_o (Tensor): (batch_dim, n_trials, x_dim)
        memory_efficient (bool, optional): If you would like to do it memory efficient or memory extensive.
            Defawults to True.
        quiet (bool, optional): remove progress bar. Defaults to False.

    Raises:
        NotImplementedError: Memory extensive approach is not implemented yet.

    Returns:
        Tensor: distance matrix (batch_dim, n_target)
    """
    if not memory_efficient:
        raise NotImplementedError(
            "Not implemented yet to do the distance matrix in memory extensive mode. But it would be way quicker."
        )
        return

    if len(x_target) > len(x_o):
        x1 = x_o
        x2 = x_target
        transpose = False
    else:
        x1 = x_target
        x2 = x_o
        transpose = True

    iterator = x1
    if not quiet:
        iterator = tqdm(
            iterator, desc=f"Compute {distance_func.__name__} distances", unit="sample"
        )

    distances = torch.stack([distance_func(x2, xs) for xs in iterator])
    if transpose:
        distances.T

    return distances
