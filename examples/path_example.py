import warnings
import numpy as np
import matplotlib.pyplot as plt
from joblib import Memory
from shcl import multitask_blockhomo_solver
from shcl.config import colors, markers
from shcl.utils import (TP_FP_FN, compute_alpha_max_blockhomo, generate_data)
from sklearn.linear_model.cd_fast import enet_coordinate_descent_multi_task


cachedir = './joblib_cache/'
memory = Memory(cachedir=cachedir, verbose=1)

generate_data = memory.cache(generate_data)


def run_solver(X, Y, solver, alphas, ax1, color, marker, true_support):
    TPs = []
    FPs = []
    FNs = []

    f_Sigma_update = 2

    n_active = len(true_support)
    Beta = np.zeros([n_features, n_tasks])

    for (ix_alpha, alpha) in enumerate(alphas):
        print("%s : alpha %d/%d" % (solver, ix_alpha, len(alphas)))
        if solver == "SCL":
            block_indices_conco = np.array([0, len(Y)]).astype(np.int32)

            Beta, sigma_invs = memory.cache(multitask_blockhomo_solver)(
                X, np.asfortranarray(Y), alpha, Beta, block_indices_conco,
                tol=1e-4,
                max_iter=20000, f_Sigma_update=f_Sigma_update,
                sigma_min=sigma_min)

        elif solver == "SCL (Block 1)":
            X1 = X[block_indices[0]:block_indices[1], :].copy(order='F')
            Y1 = Y[block_indices[0]:block_indices[1], :].copy(order='F')
            block_indices_conco1 = np.array([0, len(Y1)]).astype(np.int32)
            Beta = np.zeros([X1.shape[1], Y1.shape[1]])
            Beta, sigma_invs = memory.cache(multitask_blockhomo_solver)(
                X1, Y1, alpha, Beta, block_indices_conco1,
                tol=1e-4,
                max_iter=20000, f_Sigma_update=f_Sigma_update,
                sigma_min=sigma_min)

        elif solver == "SBHCL":
            Beta, sigma_invs = memory.cache(multitask_blockhomo_solver)(
                X, np.asfortranarray(Y), alpha, Beta, block_indices, tol=1e-4,
                max_iter=20000, f_Sigma_update=f_Sigma_update,
                sigma_min=sigma_min)

        elif solver == "MTL":
            max_iter = 20000
            tol = 1e-5
            random = 0
            # Beta = np.ascontiguousarray(Beta).T
            Beta, dual_gap_, eps_, _ = \
                memory.cache(enet_coordinate_descent_multi_task)(
                    Beta.T, alpha, 0., X, Y, max_iter, tol,
                    np.random.RandomState(0), random)
            Beta = Beta.T

            if dual_gap_ > eps_:
                warnings.warn('Objective did not converge, you might want'
                              ' to increase the number of iterations')
        elif solver == 'MTL (Block 1)':
            max_iter = 20000
            tol = 1e-5
            random = 0
            X1 = X[block_indices[0]:block_indices[1], :].copy(order='F')
            Y1 = Y[block_indices[0]:block_indices[1], :].copy(order='F')
            Beta = np.ascontiguousarray(Beta).T
            Beta, dual_gap_, eps_, _ = \
                memory.cache(enet_coordinate_descent_multi_task)(
                    Beta, alpha, 0., X1, Y1, max_iter, tol,
                    np.random.RandomState(0), random)
            Beta = Beta.T

            if dual_gap_ > eps_:
                warnings.warn('Objective did not converge, you might want'
                              ' to increase the number of iterations')
        else:
            raise ValueError("Unknown solver %s" % solver)

        support = np.where(Beta.any(axis=1))[0]
        TP, FP, FN = TP_FP_FN(true_support, support)
        TPs.append(TP)
        FNs.append(FN)
        FPs.append(FP)

    TPs, FNs, FPs = map(np.array, (TPs, FNs, FPs))

    NP = n_active  # number of positive
    NN = n_features - NP  # number of negative
    FPR = FPs / float(NN)
    TPR = TPs / float(NP)
    ax1.plot(FPR, TPR, color=color, marker=marker, label=solver)
    ax1.set_xlabel("False positive rate")
    ax1.set_ylabel("True positive rate")
    ax1.set_xlim([-0.05, 0.65])
    ax1.legend(loc="lower right")
    ax1.grid('on')

    return TPs, FNs, FPs


def run(n_samples, n_features, n_tasks, n_active, snr, rho):
    true_sigmas = [1, 2, 5]  # true noise levels
    assert n_samples % 3 == 0  # just for this example: blocks of same size
    block_sizes = np.array([n_samples // 3] * 3)
    block_indices = np.hstack([0, np.cumsum(block_sizes)]).astype(np.int32)

    n_orient = 1
    X, true_Beta, Y = \
        generate_data(n_samples, n_features, n_tasks, n_active,
                      true_sigmas, block_sizes, rho=rho, snr=snr,
                      n_orient=n_orient)

    true_support = np.where(true_Beta.any(axis=1))[0]

    sigma_min = 1e-3
    # "block homo"
    alpha_max = compute_alpha_max_blockhomo(X, Y, sigma_min, block_indices)
    n_alphas = 20
    eps = 0.3
    alpha_ratio = eps ** (1. / (n_alphas - 1))
    alphas_block_homo = alpha_max * (alpha_ratio ** np.arange(n_alphas))

    # "conco"
    block_indices_conco = np.array([0, n_samples]).astype(np.int32)
    alpha_max = compute_alpha_max_blockhomo(X, Y, sigma_min,
                                            block_indices_conco)
    n_alphas = 20
    eps = 0.3
    alpha_ratio = eps ** (1. / (n_alphas - 1))
    alphas_conco = alpha_max * (alpha_ratio ** np.arange(n_alphas))

    # "mtl"
    alpha_max = np.max(np.linalg.norm(X.T.dot(Y), axis=1))
    n_alphas = 20
    eps = 0.01
    alpha_ratio = eps ** (1. / (n_alphas - 1))
    alphas_homo = alpha_max * (alpha_ratio ** np.arange(n_alphas))

    # "mtl (block 1)"
    X1 = X[block_indices[0]:block_indices[1], :]
    Y1 = Y[block_indices[0]:block_indices[1], :]
    alpha_max = np.max(np.linalg.norm(X1.T.dot(Y1), axis=1))
    n_alphas = 20
    eps = 0.01
    alpha_ratio = eps ** (1. / (n_alphas - 1))
    alphas_lasso1 = alpha_max * (alpha_ratio ** np.arange(n_alphas))

    # "conco1"
    alpha_max = compute_alpha_max_blockhomo(X1, Y1, sigma_min,
                                            np.array([0, len(Y1)]))
    n_alphas = 20
    eps = 0.3
    alpha_ratio = eps ** (1. / (n_alphas - 1))
    alphas_conco1 = alpha_max * (alpha_ratio ** np.arange(n_alphas))

    plt.close('all')
    fig, ax1 = plt.subplots(1, 1, figsize=(6.3, 3.3))
    TPs, FNs, FPs = run_solver(X, Y, 'SBHCL', alphas_block_homo, ax1,
                               color=colors[0], marker=markers[0],
                               true_support=true_support)
    TPs, FNs, FPs = run_solver(X, Y, 'MTL', alphas_homo, ax1,
                               color=colors[1], marker=markers[1],
                               true_support=true_support)
    TPs, FNs, FPs = run_solver(X, Y, 'MTL (Block 1)', alphas_lasso1, ax1,
                               color=colors[2], marker=markers[2],
                               true_support=true_support)
    TPs, FNs, FPs = run_solver(X, Y, 'SCL', alphas_conco, ax1,
                               color=colors[3], marker=markers[3],
                               true_support=true_support)
    TPs, FNs, FPs = run_solver(X1, Y1, 'SCL (Block 1)', alphas_conco1, ax1,
                               color=colors[4], marker=markers[4],
                               true_support=true_support)


if __name__ == "__main__":
    n_samples, n_features, n_tasks = 150, 1000, 10
    n_active = 50
    rho = 0.9
    snr = 1

    block_sizes = np.array([n_samples // 3] * 3)
    block_indices = np.hstack([0, np.cumsum(block_sizes)]).astype(np.int32)
    sigma_min = 1e-6
    run(n_samples, n_features, n_tasks, n_active, snr, rho)
    plt.show()
