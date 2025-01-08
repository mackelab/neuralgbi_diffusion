import numpy as np


def get_positional_encoding(t: np.ndarray, d: int = 512, n: int=10_000) -> np.ndarray:
    """get positional encoding for a time vector t

    Args:
        t (np.ndarray): time indices. Integers >= 0 with shape (n_samples, )
        d (int, optional): dimension of the final positional encoding. Has to be an even number Defaults to 512. 
        n (int, optional): User defined scalar. Defaults to 10_000.

    Returns:
        np.ndarray: (n_samples, d)
    """
    n_samples = len(t)
    pos_enc = np.zeros((n_samples, d))
    
    pos_index = np.arange(d // 2)
    denominator = np.power(n, 2 * pos_index / d)

    pos_enc[:, 2*pos_index] = np.sin(t[:, None] / denominator[None])
    pos_enc[:, 2*pos_index + 1] = np.cos(t[:, None] / denominator[None])

    return pos_enc