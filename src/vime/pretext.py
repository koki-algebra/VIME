import numpy as np
import torch
from torch import Tensor
from typing import Tuple


def pretext_generator(X: Tensor, p_m: float, device: str) -> Tuple[Tensor, Tensor]:
    # generate mask matrix
    mask = np.random.binomial(n=1, p=p_m, size=X.size())
    M = torch.from_numpy(mask.astype(np.float32)).clone().to(device)

    # Randomly (and column-wise) shuffle data
    rows, cols = X.size()
    X_bar = torch.zeros(size=X.size()).to(device)
    for i in range(cols):
        ids = torch.randperm(rows)
        X_bar[:, i] = X[ids, i]

    # Corrupt samples
    X_tilde = X * (1 - M) + X_bar * M

    # true mask
    M = 1.0 * (X != X_tilde)

    return M, X_tilde
