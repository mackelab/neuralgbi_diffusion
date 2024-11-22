import os
from typing import Dict, List, Tuple

import numpy as np
import torch
from matplotlib.axis import Axis
from matplotlib.figure import Figure
from typing import Callable
import seaborn as sns
import pandas as pd
from tqdm import tqdm
from gbi_diff.utils.mcmc_config import Config



def plot_correlation(
    pred: torch.Tensor, d: torch.Tensor, as_array: bool = False, agg: bool = False,
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


def pair_plot_stack(samples: Dict[str, torch.Tensor], checkpoint: str, config: Config, save_dir: str = None) -> List[sns.PairGrid]:
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
        grid = pair_plot(samples["theta"][idx], potential_fn=potential_func, title=f"Index: {idx}")
        figures.append(grid)
        
        save_path = get_sample_path(checkpoint, f"pair_plot_{idx}.png", save_dir)
        directory = "/".join(save_path.split("/")[:-1])
        if not os.path.exists(directory):
            os.makedirs(directory)
        grid.savefig(save_path)
    return figures
    


def pair_plot(
    theta: torch.Tensor,
    potential_fn: Callable[[torch.Tensor], torch.Tensor] = None,
    s: int = 10,
    alpha: float = 1,
    title: str = None,
    save_path: str = None
):
    """_summary_

    Args:
        theta (torch.Tensor): (n_samples, feature_dim)
        potential_fn (str optional): potential function for log posterior. Defaults to None.
        s (int, optional): marker size. Defaults to 10.
        alpha (float, optional): maker transparency. Defaults to 1.
        title (str, optional): figure title. Defaults to None
    """
    n_feautures = theta.shape[1]
    columns = ["theta_" + str(idx) for idx in range(n_feautures)]
    plot_kws = {"s": s, "alpha": alpha}

    if potential_fn is not None:
        with torch.no_grad():
            p = torch.exp(potential_fn.log_posterior(theta))
        plot_kws["c"] = p
    df = pd.DataFrame(torch.cat([theta, p], dim=1), columns=columns + ["p"])
    grid = sns.pairplot(df, vars=columns, corner=True, plot_kws=plot_kws)
    
    if title is not None:
        grid.figure.suptitle(title)

    if save_path is not None:
        grid.savefig(save_path)

    return grid