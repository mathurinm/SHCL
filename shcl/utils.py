import numpy as np
from numpy.linalg import norm


def compute_alpha_max(X, y, sigma_min, block_indices):
    n_samples = X.shape[0]
    n_sigmas = len(block_indices) - 1
    sigmas_hat = np.zeros(n_sigmas)
    Sigma_inv_y = y.copy()

    for k in range(n_sigmas):
        block = slice(block_indices[k], block_indices[k + 1])
        bs = block_indices[k + 1] - block_indices[k]
        sigmas_hat[k] = max(sigma_min, norm(y[block]) / np.sqrt(bs))
        Sigma_inv_y[block] /= sigmas_hat[k]

    alpha_max = norm(np.dot(X.T, Sigma_inv_y), ord=np.inf) / n_samples
    return alpha_max
