import numpy as np
from numpy.linalg import norm
from scipy.linalg import toeplitz
from sklearn.utils import check_random_state


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
    return norm_l2inf(np.dot(X.T, np.dot(Sigma_max_inv, Y))
                      ) / (n_samples * n_tasks)


def compute_alpha_max_blockhomo(X, Y, sigma_min, block_indices, n_orient=1):
    n_samples, n_tasks = Y.shape
    n_groups = X.shape[1] // n_orient
    block_sizes = np.diff(block_indices)
    n_blocks = len(block_sizes)

    sigmas_max = np.zeros(n_blocks)
    for k in range(n_blocks):
        block = slice(block_indices[k], block_indices[k + 1])
        sigmas_max[k] = norm(Y[block, :], ord='fro') / \
            np.sqrt((n_tasks * block_sizes[k]))

    sigmas_max = np.maximum(sigmas_max, sigma_min)
    Sigma_max_inv = np.diag(1. / np.repeat(sigmas_max, block_sizes))
    return norm_l2inf(np.dot(X.T, np.dot(Sigma_max_inv, Y)).reshape(
        n_groups, -1)) / (n_samples * n_tasks)


def TP_FP_FN(true_support, support_hat):
    TP = len(set(true_support).intersection(support_hat))
    FP = len(support_hat) - TP
    FN = len(set(true_support) - set(support_hat))

    return TP, FP, FN


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
