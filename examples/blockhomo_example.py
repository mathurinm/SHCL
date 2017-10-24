import numpy as np
from numpy.linalg import norm
from joblib import Memory

from shcl import multitask_blockhomo_solver
from shcl.utils import (TP_FP_FN, compute_alpha_max_blockhomo)
from scipy.linalg import toeplitz
from sklearn.utils import check_random_state


cachedir = './joblib_cache/'
memory = Memory(cachedir=cachedir, verbose=1)


@memory.cache
def generate_data(n_samples, n_features, n_tasks, true_support_size,
                  true_sigmas, block_sizes,
                  rho=0.5, snr=2, random_state=24):
    """Generate Toeplitz correlated features with block homoscedastic noise
    and given signal-to-noise ratio"""
    rng = check_random_state(random_state)
    vect = rho ** np.arange(n_features)
    covar = toeplitz(vect, vect)
    X = rng.multivariate_normal(np.zeros(n_features), covar, n_samples)

    true_support = rng.choice(range(n_features), true_support_size,
                                    replace=False)
    np.sort(true_support)
    true_Beta = np.zeros([n_features, n_tasks])
    true_Beta[true_support] = rng.randn(true_support_size, n_tasks)
    Y = np.dot(X, true_Beta)

    Sigma = np.diag(np.repeat(true_sigmas, block_sizes))
    noise = np.dot(Sigma,rng.randn(n_samples, n_tasks))
    s = norm(Y) / (norm(noise) * snr)
    noise *= s

    Y += noise
    Y = np.asfortranarray(Y)

    return np.asfortranarray(X), true_Beta, Y


if __name__ == "__main__":
    n_samples, n_features, n_tasks = 60, 60, 200
    true_support_size = 20
    true_sigmas = [1, 2, 3]
    assert n_samples % 3 == 0  # just for this example: blocks of same size
    block_sizes = np.array([n_samples // 3] * 3)
    block_indices = np.hstack([0, np.cumsum(block_sizes)]).astype(np.int32)

    X, true_Beta, Y = \
        generate_data(n_samples, n_features, n_tasks, true_support_size,
                      true_sigmas, block_sizes, rho=0.5, snr=2)

    sigma_min = 1e-3

    alpha_max = compute_alpha_max_blockhomo(X, Y, sigma_min, block_indices)
    alpha = alpha_max / 1.8

    Beta_init = np.zeros([n_features, n_tasks])
    max_iter = 30000
    f_Sigma_update = 200

    Beta, sigmas_inv = \
        multitask_blockhomo_solver(X, Y, alpha, Beta_init, block_indices,
                                   max_iter=max_iter,
                                   f_Sigma_update=f_Sigma_update,
                                   sigma_min=sigma_min)

    true_support = np.where(true_Beta.any(axis=1))[0]
    support = np.where(Beta.any(axis=1))[0]

    print("TP {}, FP {}, FN {}".format(*TP_FP_FN(true_support, support)))
