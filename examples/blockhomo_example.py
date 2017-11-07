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
def generate_data(n_samples, n_features, n_tasks, n_NNZ_groups,
                  true_sigmas, block_sizes,
                  rho=0.5, snr=2, random_state=24, n_orient=1):
    """Generate Toeplitz correlated features with block homoscedastic noise
    and given signal-to-noise ratio"""
    if n_features % n_orient != 0:
        raise ValueError("Incompatible n_features and n_orient")
    n_groups = n_features // n_orient

    rng = check_random_state(random_state)
    vect = rho ** np.arange(n_features)
    covar = toeplitz(vect, vect)
    X = rng.multivariate_normal(np.zeros(n_features), covar, n_samples)

    true_support = rng.choice(range(n_groups), n_NNZ_groups,
                              replace=False)
    if n_orient != 1:
        true_support = (np.arange(n_orient)[:, None] +
                        n_orient * true_support[None, :]).ravel()
    print(true_support)

    np.sort(true_support)
    true_Beta = np.zeros([n_features, n_tasks])
    true_Beta[true_support] = rng.randn(n_orient * n_NNZ_groups, n_tasks)
    Y = np.dot(X, true_Beta)

    Sigma = np.diag(np.repeat(true_sigmas, block_sizes))
    noise = np.dot(Sigma, rng.randn(n_samples, n_tasks))
    s = norm(Y) / (norm(noise) * snr)
    noise *= s

    Y += noise
    Y = np.asfortranarray(Y)

    return np.asfortranarray(X), true_Beta, Y


if __name__ == "__main__":
    n_samples, n_features, n_tasks = 60, 60, 200
    n_NNZ_groups = 5  # number of non zero groups
    true_sigmas = [1, 2, 5]
    assert n_samples % 3 == 0  # just for this example: blocks of same size
    # block_sizes = np.array([n_samples])
    block_sizes = np.array([n_samples // 3] * 3)
    block_indices = np.hstack([0, np.cumsum(block_sizes)]).astype(np.int32)

    n_orient = 3
    X, true_Beta, Y = \
        generate_data(n_samples, n_features, n_tasks, n_NNZ_groups,
                      true_sigmas, block_sizes, rho=0.5, snr=2,
                      n_orient=n_orient)

    sigma_min = 1e-3

    alpha_max = compute_alpha_max_blockhomo(X, Y, sigma_min, block_indices,
                                            n_orient=n_orient)
    alpha = alpha_max / 1.8

    Beta_init = np.zeros([n_features, n_tasks])
    max_iter = 10000
    f_Sigma_update = 200
    tol = 1e-6

    Beta, sigmas_inv = \
        multitask_blockhomo_solver(X, Y, alpha, Beta_init, block_indices,
                                   max_iter=max_iter,
                                   f_Sigma_update=f_Sigma_update,
                                   sigma_min=sigma_min, n_orient=n_orient,
                                   tol=tol)

    true_support = np.where(true_Beta.any(axis=1))[0]
    support = np.where(Beta.any(axis=1))[0]

    print("TP {}, FP {}, FN {}".format(*TP_FP_FN(true_support, support)))
    print("sigmas:", 1. / sigmas_inv)
