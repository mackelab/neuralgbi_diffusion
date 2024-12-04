from torch import Tensor
import torch


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