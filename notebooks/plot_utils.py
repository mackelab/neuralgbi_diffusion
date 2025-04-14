import numpy as np
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from torch import Tensor

from gbi_diff.model.lit_module import Guidance
import matplotlib.colors as mcolors


def create_asymmetric_colormap(vmin, vmax, vcenter, cmap_name='coolwarm'):
    """
    Creates an asymmetric colormap by adjusting the normalization so that colors are squished based on asymmetry.
    
    Parameters:
    vmin (float): Minimum data value.
    vmax (float): Maximum data value.
    vcenter (float): Center value for asymmetry.
    cmap_name (str): Name of the base colormap.
    
    Returns:
    colormap, norm: The modified colormap and normalization.
    """
    
    # Get original colormap
    cmap = plt.get_cmap(cmap_name)
    
    # Define asymmetry
    lower_portion = (vcenter - vmin) / (vmax - vmin)
    upper_portion = (vmax - vcenter) / (vmax - vmin)
    
    # Create new color mapping
    if lower_portion < upper_portion:
        # More space above center (squish lower part)
        norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
    else:
        # More space below center (squish upper part)
        norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
    
    return cmap, norm

def plot_guidance_analytics_2D(
    samples: Tensor,
    gt_sample: Tensor,
    gt_distance: Tensor,
    pred_distance: Tensor,
    pred_grad: Tensor,
    gt_grad: Tensor = None,
    title: str = "",
):
    if gt_grad is not None:
        fig, axs = plt.subplots(
            ncols=3, nrows=2, sharex=True, sharey=True, figsize=(7, 4.3)
        )
        ((ax1, ax2, ax5), (ax3, ax4, ax6)) = axs
    else:
        fig, axs = plt.subplots(
            ncols=2, nrows=2, sharex=True, sharey=True, figsize=(3.5, 3.5)
        )
        ((ax1, ax2), (ax3, ax4)) = axs

    grad_norm = torch.linalg.norm(pred_grad, dim=-1)
    if len(pred_grad.shape) == 3:
        gt_sample = gt_sample.mean(0)
        samples = samples.mean(1)
        pred_grad = pred_grad.mean(1)
        grad_norm = grad_norm.mean(-1)

    ax1.scatter(*samples.T, c=gt_distance, s=5)
    if gt_grad is not None:
        ax1.quiver(*samples.T, *gt_grad.T)
    ax1.scatter(*gt_sample, c="r", s=10)
    ax1.set_title("GT")

    difference = pred_distance - gt_distance
    extreme = torch.max(difference[~torch.isnan(difference)].abs())
    ax2.scatter(*samples.T, c=difference - difference.mean(), cmap="bwr", vmin=-extreme, vmax=extreme, s=5)
    ax2.scatter(*gt_sample, c="k", s=10)
    ax2.set_title("Difference: Guidance - GT")

    ax3.scatter(*samples.T, c=pred_distance, s=5)
    ax3.quiver(*samples.T, *pred_grad.T)
    ax3.scatter(*gt_sample, c="r", s=10)
    ax3.set_title("Guidance")

    ax4.scatter(*samples.T, c=grad_norm, cmap="inferno", s=5)
    ax4.scatter(*gt_sample, c="r", s=10)
    ax4.set_title("Gradient Magnitude")

    if gt_grad is not None:
        gt_grad_norm = torch.linalg.norm(gt_grad, dim=-1)
        difference = gt_grad_norm - grad_norm
        vmin = torch.min(difference[~torch.isnan(difference)].abs())
        vmax = torch.max(difference[~torch.isnan(difference)].abs())
        cmap, norm = create_asymmetric_colormap(vmin, vmax, 0, 'coolwarm')
        ax5.scatter(*samples.T, c=difference, cmap=cmap, vmin=vmin, vmax=vmax, norm=norm, s=5)
        ax5.scatter(*gt_sample, c="k", s=10)
        ax5.set_title("Difference: L2(GT grad) - L2(Guidance grad)")

        cos_similarity = torch.sum(gt_grad * pred_grad, dim=-1) / gt_grad_norm / grad_norm
        extreme = torch.max(cos_similarity[~torch.isnan(cos_similarity)].abs())
        ax6.scatter(*samples.T, c=cos_similarity, cmap="bwr", vmin=-extreme, vmax=extreme, s=5)
        ax6.scatter(*gt_sample, c="k", s=10)
        ax6.set_title("Grad Cos Similarity")
    
    fig.suptitle(title)
    fig.tight_layout()
    return fig, axs


def plot_guidance_analytics_2D_minimal(
    samples: Tensor,
    gt_sample: Tensor,
    gt_distance: Tensor,
    pred_distance: Tensor,
    pred_grad: Tensor,
    gt_grad: Tensor = None,
    title: str = "",
):
    fig, axs = plt.subplots(
        ncols=2, nrows=2, sharex=True, sharey=True, figsize=(4, 4)
    )
    ((ax1, ax2), (ax3, ax4)) = axs

    grad_norm = torch.linalg.norm(pred_grad, dim=-1)
    if len(pred_grad.shape) == 3:
        gt_sample = gt_sample.mean(0)
        samples = samples.mean(1)
        pred_grad = pred_grad.mean(1)
        grad_norm = grad_norm.mean(-1)

    ax1.scatter(*samples.T, c=gt_distance, s=5)
    if gt_grad is not None:
        ax1.quiver(*samples.T, *gt_grad.T)
    ax1.scatter(*gt_sample, c="r", s=10)
    ax1.set_title("GT")

    difference = pred_distance - gt_distance
    extreme = torch.max(difference[~torch.isnan(difference)].abs())
    ax2.scatter(*samples.T, c=difference - difference.mean(), cmap="bwr", vmin=-extreme, vmax=extreme, s=5)
    ax2.scatter(*gt_sample, c="k", s=10)
    ax2.set_title("Difference: Guidance - GT")

    ax3.scatter(*samples.T, c=pred_distance, s=5)
    ax3.quiver(*samples.T, *pred_grad.T)
    ax3.scatter(*gt_sample, c="r", s=10)
    ax3.set_title("Guidance")

    
    gt_grad_norm = torch.linalg.norm(gt_grad, dim=-1)
    difference = gt_grad_norm - grad_norm
    vmin = torch.min(difference[~torch.isnan(difference)].abs())
    vmax = torch.max(difference[~torch.isnan(difference)].abs())
    cmap, norm = create_asymmetric_colormap(vmin, vmax, 0, 'coolwarm')
    ax4.scatter(*samples.T, c=difference, cmap=cmap, vmin=vmin, vmax=vmax, norm=norm, s=5)
    ax4.scatter(*gt_sample, c="k", s=10)
    ax4.set_title("Difference: L2(GT grad) - L2(Guidance grad)")
    
    fig.suptitle(title)
    fig.tight_layout()
    return fig, axs




def plot_guidance_analytics_1D(
    samples: Tensor,
    gt_sample: Tensor,
    gt_distance: Tensor,
    pred_distance: Tensor,
    pred_grad: Tensor,
    gt_grad: Tensor = None,
    title: str = "",
):
    if gt_grad is not None:
        fig, axs = plt.subplots(
            ncols=3, nrows=2, sharex=True, figsize=(7, 4.3)
        )
        ((ax1, ax2, ax6), (ax3, ax4, ax5)) = axs
    else:
        fig, axs = plt.subplots(
            ncols=2, nrows=2, sharex=True, figsize=(3.5, 3.5)
        )
        ((ax1, ax2), (ax3, ax4)) = axs

    grad_norm = torch.linalg.norm(pred_grad, dim=-1)

    ax1.scatter(samples, gt_distance, c=gt_distance, s=4)
    if gt_grad is not None:
        ax1.quiver(samples, gt_distance, gt_grad, torch.zeros_like(pred_distance))
    ax1.axvline(gt_sample, c="r")
    ax1.set_title("GT")

    ax3.scatter(samples, pred_distance, c=pred_distance, s=4)
    ax3.quiver(samples, pred_distance, pred_grad, torch.zeros_like(pred_distance))
    ax3.axvline(gt_sample, c="r")
    ax3.set_title("Guidance")

    difference = pred_distance - gt_distance
    extreme = difference.abs().max()
    ax2.scatter(samples, difference, c=difference, cmap="bwr", vmin=-extreme, vmax=extreme)
    ax2.axvline(gt_sample, c="k")
    ax2.set_title("Difference: Guidance - GT")

    ax4.scatter(samples, grad_norm, c=grad_norm, cmap="inferno")
    ax4.axvline(gt_sample, c="k")
    ax4.set_title("Gradient Magnitude")

    if gt_grad is not None:
        sort_idx = torch.argsort(samples.flatten())
        ax4.plot(
            samples.flatten()[sort_idx], gt_grad.flatten().abs()[sort_idx],
            "k--",
        )
        difference =  pred_grad - gt_grad
        extreme = difference.abs().max()
        ax5.scatter(samples, difference, c=difference, cmap="bwr", vmin=-extreme, vmax=extreme)
        ax5.set_title("Difference: L2(gt grad) - L2(pred grad)")
    
        ax6.plot(samples.flatten()[sort_idx], gt_distance.flatten()[sort_idx], label="GT")
        ax6.plot(samples.flatten()[sort_idx], pred_distance.flatten()[sort_idx], label="Guidance")        
        ax6.legend()
        ax6.axvline(gt_sample, c="k")
        ax6.set_title("Comparison: GT, Guidance")

    ax2.set_xlabel("Samples")
    ax4.set_xlabel("Samples")
    fig.suptitle(title)
    fig.tight_layout()
    return fig, axs

def get_guidance_analytics(theta: Tensor, x: Tensor, guidance: Guidance, t: int = 0):
    """_summary_

    Args:
        theta (Tensor): (batch_dim, theta_dim)
        x (Tensor): (n_target, x_dim)
        guidance (Guidance): guidance function
        t (int, optional): diffusion time in index space. Defaults to 0.
    """
    time_repr = guidance.get_diff_time_repr(np.ones(len(theta)) * t)
    theta = theta.detach()
    theta.requires_grad = True
    x = x.detach()
    x.requires_grad = True
    time_repr = time_repr.detach()
    time_repr.requires_grad = True
    repeat_seq = np.ones(len(x.shape) + 1, dtype=int)
    repeat_seq[0] = len(theta)

    pred_dist = guidance.forward(theta, x[None].repeat(*repeat_seq), time_repr)
    grad = torch.autograd.grad(outputs=pred_dist.sum(), inputs=(theta, x))
    theta = theta.detach()
    x = x.detach()
    pred_dist = pred_dist.detach()
    grad = [g.detach() for g in grad]

    return pred_dist, grad


def plot_corr(gt_distance, pred_dist, gradient=None):
    if gradient is not None:
        grad_norm = torch.linalg.norm(gradient, dim=-1)
        if len(grad_norm.shape) == 2:
            grad_norm = grad_norm.mean(-1)
        stack = torch.stack(
            [
                gt_distance,
                pred_dist.squeeze(),
                gt_distance - pred_dist.squeeze(),
                grad_norm,
            ]
        )
        labels = ["gt distance", "pred distance", "distance difference", "grad norm"]
    else:
        stack = torch.stack(
            [gt_distance, pred_dist.squeeze(), gt_distance - pred_dist.squeeze()]
        )
        labels = ["gt distance", "pred distance", "distance difference", "grad norm"]

    corr = torch.corrcoef(stack)
    fig, ax = plt.subplots(figsize=(4, 4))
    sns.heatmap(
        corr,
        vmax=1,
        vmin=-1,
        cmap="coolwarm",
        annot=True,
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_aspect("equal")
