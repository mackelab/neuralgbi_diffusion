import os
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
import torch
from einops import rearrange
from matplotlib.axes import Axes
from matplotlib.axis import Axis
from matplotlib.figure import Figure
from tqdm import tqdm

from gbi_diff.utils.sampling_mcmc_config import Config
from gbi_diff.utils.metrics import batch_correlation


def plot_correlation(
    pred: torch.Tensor,
    d: torch.Tensor,
    as_array: bool = False,
    agg: bool = False,
) -> Tuple[Figure, Axis] | np.ndarray:
    """plot correlation

    Args:
        pred (torch.Tensor): distance predictions (n_samples, n_target)
        d (torch.Tensor): actual distances (n_samples, n_target)
        as_array (bool, optional): If you would like to return an array. Defaults to False.

    Returns:
        Tuple[Figure, Axis] | np.ndarray: the figure in different formats
    """

    if agg:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

    corr = torch.corrcoef(torch.stack([d.flatten(), pred.flatten()]))[0, 1]
    fig, ax = plt.subplots()

    maximum = max(torch.max(d), torch.max(pred))
    ax.plot(
        [0, maximum * 1.05],
        [0, maximum * 1.05],
        "k--",
        alpha=0.5,
    )
    rss = torch.mean((d - pred) ** 2)  # x - y
    ax.scatter(d.flatten(), pred.flatten(), s=10)
    ax.set_title(
        f"Cost estimation\nbatch-size={len(d)}, n-target={d.shape[1]}\ncorr={corr:.2}, RSS={rss:.2}"
    )
    ax.set_xlabel("true costs")
    ax.set_ylabel("predicted costs")

    if as_array:
        fig.canvas.draw()
        buf = fig.canvas.tostring_rgb()
        # Convert the canvas to a raw RGB buffer
        ncols, nrows = fig.canvas.get_width_height()
        image = np.frombuffer(buf, dtype=np.uint8).reshape(nrows, ncols, 3)
        return image

    return fig, ax


def plot_diffusion_step_loss(
    pred: torch.Tensor, target: torch.Tensor, agg: bool = False
) -> Tuple[Figure, Axes]:
    """plot squared error over diffusion time

    Args:
        pred (Tensor): (batch_size, n_target)
        target (Tensor): (batch_size, n_target)

    Returns:
        Tuple[Figure, Axes]:
    """

    if agg:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

    diff = torch.square(pred - target[:, None])
    # diff = rearrange(diff, "B D T -> (B T) D")

    fig, ax = plt.subplots()

    mean = diff.mean(dim=0)
    std = diff.std(dim=0)
    x = torch.arange(pred.shape[1])

    ax.fill_between(x, mean - std, mean + std, alpha=0.6, label="std")
    ax.plot(diff.mean(dim=0), label="mean")
    ax.legend()

    ax.set_ylabel(r"Squared Loss.")
    ax.set_xlabel("Diffusion steps")
    ax.set_title("Squared Loss over Diffusion Steps")

    return fig, ax


def plot_diffusion_step_corr(
    pred: torch.Tensor, target: torch.Tensor, agg: bool = False
) -> Tuple[Figure, Axes]:
    """plot correlation with target over diffusion time

    Args:
        pred (Tensor): (batch_size, n_diffusion_steps, n_target)
        target (Tensor): (batch_size, n_target)

    Returns:
        Tuple[Figure, Axes]:
    """

    if agg:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

    batch_size = pred.shape[0]
    n_diff_steps = pred.shape[1]
    res = torch.empty((batch_size, n_diff_steps))
    for diff_idx in range(n_diff_steps):
        res[:, diff_idx] = batch_correlation(pred[:, diff_idx], target)

    mean = res.mean(dim=0)
    std = res.std(dim=0)
    fig, ax = plt.subplots()

    x = torch.arange(n_diff_steps)

    ax.fill_between(x, mean - std, mean + std, alpha=0.6, label="std")
    ax.plot(mean, label="mean")
    ax.legend()

    ax.set_ylim([-0.05, 1.05])
    ax.set_ylabel(r"Correlation")
    ax.set_xlabel("Diffusion steps")
    ax.set_title("Correlation with target over Diffusion Steps")
    return fig, ax


def pair_plot_stack_potential(
    samples: Dict[str, torch.Tensor],
    checkpoint: str,
    config: Config,
    save_dir: str = None,
) -> List[sns.PairGrid]:
    """_summary_

    Args:
        samples (Dict[str, torch.Tensor]): _description_
        checkpoint (str): _description_
        config (Config): _description_
        save_dir (str, optional): _description_. Defaults to None.

    Returns:
        List[sns.PairGrid]: _description_
    """
    # avoid circular import
    from gbi_diff.sampling.utils import create_potential_fn, get_sample_path

    potential_func = create_potential_fn(checkpoint, config)

    figures = []
    n_samples = samples["theta"].shape[0]
    for idx in tqdm(range(n_samples), desc="Pair plot"):
        potential_func.update_x_o(samples["x_o"][idx])
        grid = pair_plot_potential(
            samples["theta"][idx], potential_fn=potential_func, title=f"Index: {idx}"
        )
        figures.append(grid)

        save_path = get_sample_path(checkpoint, f"pair_plot_{idx}.png", save_dir)
        directory = "/".join(save_path.split("/")[:-1])
        if not os.path.exists(directory):
            os.makedirs(directory)
        grid.savefig(save_path)
    return figures


def pair_plot_potential(
    theta: torch.Tensor,
    potential_fn: Callable[[torch.Tensor], torch.Tensor] = None,
    s: int = 10,
    alpha: float = 1,
    title: str = None,
    save_path: str = None,
):
    """_summary_

    Args:
        theta (torch.Tensor): (n_samples, feature_dim)
        potential_fn (str optional): potential function for log posterior. Defaults to None.
        s (int, optional): marker size. Defaults to 10.
        alpha (float, optional): maker transparency. Defaults to 1.
        title (str, optional): figure title. Defaults to None
    """
    c = None
    if potential_fn is not None:
        with torch.no_grad():
            c = torch.exp(potential_fn.log_posterior(theta))

    grid = _pair_plot(theta, c, s, alpha, title, save_path)
    return grid


def _pair_plot(
    theta: torch.Tensor,
    c: torch.Tensor = None,
    s: int = 10,
    alpha: float = 1,
    title: str = None,
    save_path: str = None,
):
    n_feautures = theta.shape[1]
    columns = ["theta_" + str(idx) for idx in range(n_feautures)]
    df = pd.DataFrame(theta.detach().numpy(), columns=columns)

    plot_kws = {"s": s, "alpha": alpha}
    if c is not None:
        plot_kws["c"] = c.detach().numpy()
    grid = sns.pairplot(df, vars=columns, corner=True, plot_kws=plot_kws)

    if title is not None:
        grid.figure.suptitle(title)

    if save_path is not None:
        save_path: Path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        grid.savefig(save_path)

    return grid
