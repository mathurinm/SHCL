import numpy as np
import time
cimport numpy as np
from scipy.linalg.cython_blas cimport ddot, dasum, daxpy, dnrm2, dcopy
from libc.math cimport fabs, sqrt, ceil
from libc.stdlib cimport rand, srand
cimport cython


cdef inline double fmax(double x, double y) nogil:
    return x if x > y else y


cdef inline double fmin(double x, double y) nogil:
    return y if x > y else y


cdef inline double fsign(double x) nogil :
    if x == 0.:
        return 0.
    elif x > 0.:
        return 1.
    else:
        return - 1.


cdef inline double ST(double u, double x) nogil:
    return fsign(x) * fmax(fabs(x) - u, 0.)


cdef double abs_max(int n, double * a) nogil:
    cdef int ii
    cdef double m = 0.
    cdef double d
    for ii in range(n):
        d = fabs(a[ii])
        if d > m:
            m = d
    return m


def solver(double[::1, :] X,
           double[:] y,
           double[:] beta,
           double alpha,
           int[:] block_indices,
           double[:] sigma_mins,
           double tol=1e-4,
           int max_iter=10 ** 5,
           int screening=0,
           int f_gap=10,
           int verbose=1):
    """
        WARNING: beta is modified by this function

        Solves the Smooth Block Homosecedastic Concomitant Lasso.
        Restricted to the diagonal covariance case for now.
    """
    cdef int n_samples = X.shape[0]
    cdef int n_features = X.shape[1]
    cdef int n_sigmas = len(block_indices) - 1

    cdef int j
    cdef int k
    cdef double sigma_k
    cdef int n_k
    cdef int s_k
    cdef double ST_const
    cdef double grad
    cdef int t
    cdef int inc = 1
    cdef double tmp
    cdef double beta_j_old
    cdef double p_obj
    cdef double d_obj
    cdef double dual_const
    cdef double highest_d_obj = 0.
    cdef double[:] theta = np.empty(n_samples)
    cdef int[:] block_sizes = np.diff(block_indices)


    cdef double[:] norm_X_cols = np.empty(n_features)
    for j in range(n_features):
        norm_X_cols[j] = dnrm2(&n_samples, &X[0, j], &inc)

    cdef double[:] R = np.empty(n_samples)
    dcopy(&n_samples, &y[0], &inc, &R[0], &inc)
    for j in range(n_features):
        if beta[j] != 0.:
            tmp = - beta[j]
            daxpy(&n_samples, &tmp, &X[0, j], &inc, &R[0], &inc)

    # block norms squared of residuals
    cdef double[:] norm_R_blocks2 = np.empty(n_sigmas)
    for k in range(n_sigmas):
        s_k = block_indices[k]
        n_k = block_sizes[k]
        norm_R_blocks2[k] = dnrm2(&n_k, &R[s_k], &inc) ** 2

    cdef int [:] disabled = np.zeros(n_features, dtype=np.int32)

    cdef double[:] sigmas = np.empty(n_sigmas)
    cdef double[::1, :] Lc = np.empty([n_sigmas, n_features], order='F')  # stands for Lipschitz constant
    cdef double[:] Sigma_inv_R = np.empty(n_samples)
    # used for clever sigma_k update
    cdef double[:] XtRj_block = np.empty(n_sigmas)
    cdef double norm2_theta_sigma_min = 0
    cdef double trace_sigma_min = 0
    for k in range(n_sigmas):
        trace_sigma_min += block_sizes[k] * sigma_mins[k]

    with nogil:
    # if 1:
        for k in range(n_sigmas):
            s_k = block_indices[k]
            n_k = block_sizes[k]
            sigmas[k] = dnrm2(&n_k, &y[s_k], &inc) / sqrt(n_k)

            for j in range(n_features):
                Lc[k, j] = dnrm2(&n_k, &X[block_indices[k], j], &inc) ** 2

        for t in range(max_iter):
            if t % f_gap == 1:
                # compute primal obj and Sigma^-1 R
                p_obj = 0.
                for k in range(n_sigmas):
                    sigma_k = sigmas[k]
                    n_k = block_sizes[k]
                    s_k = block_indices[k]
                    p_obj += dnrm2(&n_k, &R[s_k], &inc) ** 2 / (2. * sigma_k)
                    p_obj += n_k * sigma_k / 2.

                    for i in range(s_k, s_k + n_k):
                        Sigma_inv_R[i] = R[i] / sigma_k

                p_obj /= n_samples
                p_obj += alpha * dasum(&n_features, &beta[0], &inc)



                norm_XtSigma_inv_R = 0.
                for j in range(n_features):
                    norm_XtSigma_inv_R = fmax(norm_XtSigma_inv_R,
                                               fabs(ddot(&n_samples, &X[0, j], &inc,
                                                         &Sigma_inv_R[0], &inc)))
                dual_const = fmax(alpha * n_samples,
                                  dnrm2(&n_samples, &Sigma_inv_R[0], &inc) * alpha * sqrt(n_samples))
                dual_const = fmax(dual_const, norm_XtSigma_inv_R)

                for i in range(n_samples):
                    theta[i] = Sigma_inv_R[i] / dual_const

                norm2_theta_sigma_min = 0.
                for k in range(n_sigmas):
                    n_k = block_sizes[k]
                    s_k = block_indices[k]
                    for i in range(s_k, s_k + n_k):
                        norm2_theta_sigma_min += theta[i] ** 2 * sigma_mins[k]

                d_obj = alpha * ddot(&n_samples, &y[0], &inc, &theta[0], &inc) + \
                    (trace_sigma_min - n_samples ** 2 * alpha ** 2 *
                     norm2_theta_sigma_min) / (2. * n_samples)

                highest_d_obj = fmax(d_obj, highest_d_obj)
                gap = p_obj - highest_d_obj

                # if 1:
                with gil:
                    if verbose:
                        print("Iteration %d" % t)
                        print("Primal {:.10f}".format(p_obj))
                        print("Dual {:.10f}".format(highest_d_obj))
                        print("Log gap %.2e" % gap)

                    if gap < tol:
                        print("Early exit, gap: %.2e < %.2e" % (gap, tol))
                        break


            for j in range(n_features):
                if disabled[j]:
                    continue

                beta_j_old = beta[j]

                if beta[j] != 0.:
                    # R += beta[j] * X[:, j]
                    daxpy(&n_samples, &beta[j], &X[0, j], &inc,
                          &R[0], &inc)

                grad = 0.
                ST_const = 0.
                for k in range(n_sigmas):
                    n_k = block_sizes[k]
                    s_k = block_indices[k]
                    # this is not exactly XtRj since we juste modified R
                    XtRj_block[k] = ddot(&n_k, &X[s_k, j], &inc, &R[s_k], &inc)
                    grad += XtRj_block[k] / sigmas[k]

                    # add the missing term to XtRj[k]:
                    if beta_j_old != 0.:
                        XtRj_block[k] -= beta_j_old * Lc[k, j]

                    ST_const += Lc[k, j] / sigmas[k]

                beta[j] = ST(n_samples * alpha, grad) / ST_const


                if beta[j] != 0.:
                    # R -= beta[j] * X[:, j]
                    tmp = - beta[j]
                    daxpy(&n_samples, &tmp, &X[0, j], &inc,
                          &R[0], &inc)

                # update sigmas:
                tmp = beta_j_old - beta[j]
                if tmp != 0.:
                    for k in range(n_sigmas):
                        norm_R_blocks2[k] += (2. * tmp * XtRj_block[k] +
                                              tmp ** 2 * Lc[k, j])
                        # it is possible that the residuals are 0, and du to
                        # numerical errors ||R^k||^2 < 0, resulating in nan
                        # when sqrt is taken. We avoid this:
                        norm_R_blocks2[k] = max(norm_R_blocks2[k], 0.)
                        n_k = block_sizes[k]
                        sigmas[k] = fmax(sigma_mins[k],
                                         sqrt(norm_R_blocks2[k] / n_k))


    return beta, sigmas
