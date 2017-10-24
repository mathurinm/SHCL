import numpy as np
from numpy.linalg import norm


def mahalanobis_norm(A, S):
    """Return S-weighted Mahalanobis norm of A"""
    return np.sqrt((A * np.dot(S, A)).sum())


def trace(A):
    """Trace of square 2D array A"""
    n = A.shape[0]
    return np.sum(A.flat[::n + 1])


def BST(x, tau):
    """In place block soft-thresholding of x at level tau."""

    shrink = 1 - tau / norm(x)
    if shrink <= 0.:
        # return np.zeros_like(x)
        x.fill(0)
    else:
        # return shrink * x
        x *= shrink


def norm_l21(Beta):
    """Row wise L2/1 mixed norm of 2D array Beta"""
    return np.sum(norm(Beta, axis=1))


def norm_l2inf(Beta):
    """Row wise l2/inf mixed norm of 2D array Beta"""
    return np.max(norm(Beta, axis=1))


def compute_alpha_max_generalized(X, Y, sigma_min):
    n_samples, n_tasks = Y.shape
    ZZtop = np.dot(Y, Y.T) / n_tasks
    lambdas, U = np.linalg.eigh(ZZtop)
    lambdas = np.maximum(lambdas, 0.)

    mus = np.maximum(np.sqrt(lambdas), sigma_min)
    Sigma_max_inv = np.dot(U, 1. / mus[:, None] * U.T)
    return norm_l2inf(np.dot(X.T, np.dot(Sigma_max_inv, Y))) / (n_samples * n_tasks)


def compute_alpha_max_blockhomo(X, Y, sigma_min, block_indices):
    n_samples, n_tasks = Y.shape
    block_sizes = np.diff(block_indices)
    n_blocks = len(block_sizes)

    sigmas_max = np.zeros(n_blocks)
    for k in range(n_blocks):
        block = slice(block_indices[k], block_indices[k + 1])
        sigmas_max[k] = norm(Y[block, :], ord='fro') / np.sqrt((n_tasks * block_sizes[k]))

    sigmas_max = np.maximum(sigmas_max, sigma_min)
    Sigma_max_inv = np.diag(1. / np.repeat(sigmas_max, block_sizes))
    return norm_l2inf(np.dot(X.T, np.dot(Sigma_max_inv, Y))) / (n_samples * n_tasks)


def TP_FP_FN(true_support, support_hat):
    TP = len(set(true_support).intersection(support_hat))
    FP = len(support_hat) - TP
    FN = len(set(true_support) - set(support_hat))

    return TP, FP, FN
