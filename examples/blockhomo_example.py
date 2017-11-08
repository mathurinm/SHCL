import numpy as np
from joblib import Memory
from shcl import multitask_blockhomo_solver
from shcl.utils import (TP_FP_FN, compute_alpha_max_blockhomo, generate_data)

cachedir = './joblib_cache/'
memory = Memory(cachedir=cachedir, verbose=1)

generate_data = memory.cache(generate_data)

if __name__ == "__main__":
    # Generate data:
    n_samples, n_features, n_tasks = 60, 150, 200
    n_NNZ_groups = 5  # number of non zero groups in true regression parameters
    true_sigmas = [1, 2, 5]  # true noise levels
    assert n_samples % 3 == 0  # just for this example: blocks of same size
    block_sizes = np.array([n_samples // 3] * 3)
    block_indices = np.hstack([0, np.cumsum(block_sizes)]).astype(np.int32)

    n_orient = 3
    X, true_Beta, Y = \
        generate_data(n_samples, n_features, n_tasks, n_NNZ_groups,
                      true_sigmas, block_sizes, rho=0.5, snr=2,
                      n_orient=n_orient)

    # Solve problem:
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
