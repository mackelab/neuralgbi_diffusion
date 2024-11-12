from typing import Tuple
from arviz import r2_score
from matplotlib.axis import Axis
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_correlation(
    pred: torch.Tensor, d: torch.Tensor, as_array: bool = False
) -> Tuple[Figure, Axis] | np.ndarray:
    """plot correlation

    Args:
        pred (torch.Tensor): distance predictions (n_samples, n_target)
        d (torch.Tensor): actual distances (n_samples, n_target)
        as_array (bool, optional): If you would like to return an array. Defaults to False.

    Returns:
        Tuple[Figure, Axis] | np.ndarray: the figure in different formats
    """
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
