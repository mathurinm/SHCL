import numpy as np
cimport numpy as np
from scipy.linalg.cython_blas cimport (ddot, dcopy, dscal, dgemm, dger, dnrm2,
                                       dasum, daxpy)
from scipy.linalg.cython_lapack cimport dsyev
cimport cython
from libc.math cimport sqrt


cdef inline double fmax(double x, double y) nogil:
    return x if x > y else y


@cython.boundscheck(False)
@cython.cdivision(True)
cpdef my_eigh(double[::1, :] A, double[:] eigvals, int A_dim,
              double[:] WORK):
    """Compute the eigen value decomposition of the Hermitian matrix A inplace.

    Parameters:
    -----------
        A : Fortran ndarray, shape (A_dim, A_dim)
            Matrix to perform EVD on. Is overwritten to contain eigvecs on
            output.
        eigvals : ndarray, shape (A_dim,)
            Array which will contains A's eigvenvalues.
        A_dim : int
            A's first and second dimension.
        WORK : ndarray, shape (3 * A_dim - 1,)
            Array passed as memory buffer for the LAPACK routines. (these
            routines do not allocate memory themselves)
    """

    cdef char U_char = 'U'
    cdef char V_char = 'V'
    cdef int INFO  # not used but need by dsyev

    # WORK is an array of length LWORK, used as memory space by LAPACK
    cdef int LWORK = 3 * A_dim - 1
    dsyev(&V_char, &U_char, &A_dim, &A[0, 0], &A_dim, &eigvals[0],
          &WORK[0], &LWORK,
          &INFO)


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef double spectral_norm_group(double[::1, :] X, double[::1, :] Sigma_inv_X,
                                int g, double[::1, :] XgSigma_invXg,
                                double[:] L_const_eigvals,
                                double[:] L_const_WORK, int n_orient):
    """Compute spectral norm of the j-th group of features.

    Parameters:
    -----------
        X : Fortran ndarray, shape (n_samples, n_features)
            Design matrix.
        Sigma_inv_X : Fortran ndarray, shape (n_samples, n_features)
            Inverse of sqrt noise covariance matrix times design matrix.
        g : int
            Group index.
        XgSigma_invXg : ndarray, shape (n_orient, n_orient)
            Placeholder array.
        L_const_eigvals : ndarray, shape (n_orient,)
            Placeholder array to contain the eigenvalues of XgSigma_invXg.
        n_orient : int
            Number of orientation. If it is 1, using this function is overkill.
        L_const_WORK : ndarray, shape (3 * n_orient - 1)
            Memory buffer for LAPACK.

    Returns:
        The spectral norm of XgSigma_invXg.
    """
    cdef int i, k
    cdef int n_samples = X.shape[0]
    cdef int inc = 1
    for i in range(n_orient):
        for k in range(n_orient):
            XgSigma_invXg[i, k] = ddot(&n_samples, &X[0, g + i], &inc,
                                       &Sigma_inv_X[0, g + k], &inc)
            # if k != i:
            #     XgSigma_invXg[k, i] = XgSigma_invXg[i, k]

    # perform eigval dec of XgSigma_invXg
    my_eigh(XgSigma_invXg, L_const_eigvals, n_orient, L_const_WORK)
    return L_const_eigvals[n_orient - 1]


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef double spectral_norm_group_clever(double[::1, :] X, double[:] sigmas_inv,
                                int[:] block_indices, int[:] block_sizes,
                                int g, double[::1, :] XgSigma_invXg,
                                double[:] L_const_eigvals,
                                double[:] L_const_WORK, int n_orient):
    cdef int n_blocks = sigmas_inv.shape[0]
    cdef int ii, jj, k
    cdef int inc = 1
    for ii in range(n_orient):
        for jj in range(n_orient):
            XgSigma_invXg[ii, jj] = 0.
            for k in range(n_blocks):
                XgSigma_invXg[ii, jj] += \
                    ddot(&block_sizes[k], &X[block_indices[k], g + ii], &inc,
                         &X[block_indices[k], g + jj], &inc) * sigmas_inv[k]

    # perform eigval dec of XgSigma_invXg
    my_eigh(XgSigma_invXg, L_const_eigvals, n_orient, L_const_WORK)
    return L_const_eigvals[n_orient - 1]


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef BST(double * x, int x_len, double tau, double * zeros_like_x,
         int n_orient):
    """
    Block soft-thresholding of vector x at level tau, performed inplace.
    Math formula: BST(x, tau) = x * max(0., 1 - tau/||x||_F)

    Parameters:
    -----------
        x : ndarray, shape (n_orient, x_len)
    """
    cdef int inc = 1
    cdef int tmpint = n_orient * x_len  # total number of elements in x
    cdef double x_norm = dnrm2(&tmpint, x, &inc)
    if x_norm == 0.:
        # fill x with 0 with BLAS routine rather than for loop:
        dcopy(&tmpint, zeros_like_x, &inc, x, &inc)
        return

    cdef double shrink = 1. - tau / x_norm
    if shrink <= 0.:
        dcopy(&tmpint, zeros_like_x, &inc, x, &inc)
    else:
        dscal(&tmpint, &shrink, x, &inc)


@cython.cdivision(True)
@cython.boundscheck(False)
cpdef multitask_generalized_solver(double[::1, :] X, double[::1, :] Y, double alpha,
                            double[:, ::1] Beta_init, double tol=1e-4,
                            long max_iter=10**5, int f_Sigma_update=10,
                            double sigma_min=1e-3, int verbose=1, int n_orient=1):
    """Solve the optimization problem of the multi-task Generalized Smoothed
    Concomitant Lasso with alternate minimization.
    We minimise in Beta and Sigma:
    (1/(2 n_samples n_tasks) trace((Y - X Beta).T Sigma^-1 (Y - X Beta)) +
    trace(Sigma) / (2 n_samples) + alpha sum(norm(Beta, ord=2, axis=1))

    Parameters:
    -----------
        X : Fortran ndarray, shape (n_samples, n_features)
            Training data.
        Y : Fortran ndarray, shape (n_samples, n_tasks)
            Target values.
        alpha : float
            Regularization parameter.
        Beta_init : ndarray, shape (n_features, n_tasks)
            Initialization of the regression coefficients.
        tol : float
            The solver stops when reaching a duality gap lower than tol.
        max_iter : int
            If the duality gap does go below tol after max_iter epochs of
            coordinate descent, the solver stops.
        f_Sigma_update : int
            Number of coordinate descent epochs between updates of Sigma.
        sigma_min : float
            Lower constraint on the eigenvalues of Sigma.
        verbose : {0, 1}
            Level of verbosity.
        n_orient : {1, 3}
            Use only 1 except when dealing with MEEG, in which case 1 is fixed
            orientation as 3 is free orientation.
    Returns:
    --------
        Beta : ndarray, shape (n_features, n_tasks)
            Estimated regression coefficients after optimization.
        Sigma : ndarray, shape (n_samples, n_samples)
            Estimated square root of noise covariance after optimization.
    """
    cdef int n_samples = X.shape[0]
    cdef int n_features = X.shape[1]
    cdef int n_groups = n_features / n_orient
    cdef int n_tasks = Y.shape[1]
    cdef double inv_n_tasks = 1. / n_tasks

    cdef int it
    cdef int i
    cdef int j
    cdef int g  # groups. If n_orient = 1, groups are single features
    cdef int inc = 1
    cdef double zero = 0.
    cdef double one = 1.  # to pass to cblas
    cdef double minus_one = - 1.  # for cblas
    cdef int tmpint
    cdef double tmpdouble
    cdef double dual_norm_XtTheta

    cdef char trans_id = 'n'
    cdef char trans_t = 't'

    # Lapack magic
    cdef int LWORK = 3 * n_samples - 1
    cdef double[:] WORK = np.empty(LWORK)

    # matrix we will use to store the eigevectors of ZZtop
    cdef double[::1, :] U = np.empty([n_samples, n_samples], order='F')

    # Lispchitz constants of features
    cdef double[:] L_const = np.empty(n_features)
    cdef double[:, ::1] zeros_like_Beta_g = np.zeros([n_orient, n_tasks])
    # TODO monitor Energies as list
    cdef double[::1, :] R = np.empty([n_samples, n_tasks], order='F')
    cdef double[::1, :] ZZtop = np.empty([n_samples, n_samples], order='F')

    tmpint = n_features * n_tasks
    cdef double[:, :] Beta = np.empty([n_features, n_tasks])
    dcopy(&tmpint, &Beta_init[0, 0], &inc, &Beta[0, 0], &inc)

    cdef double[::1, :] Sigma_inv = np.empty([n_samples, n_samples], order='F')
    cdef double[::1, :] Sigma_inv_R = np.empty([n_samples, n_tasks], order='F')
    cdef double[::1, :] Sigma_inv_X = np.empty([n_samples, n_features], order='F')
    cdef double[::1, :] zeros_like_Sigma = np.zeros([n_samples, n_samples], order='F')
    cdef double[::1, :] Theta = np.zeros([n_samples, n_tasks], order='F')
    cdef double [:] XjTheta = np.zeros(n_tasks)
    cdef double[:] lambdas = np.empty(n_samples)
    cdef double[:] mus = np.empty(n_samples)
    cdef double[:] invmus = np.empty(n_samples)

    cdef double p_obj
    cdef double d_obj
    cdef double gap
    cdef double scal
    cdef double norm2Theta

    # cdef double[::1, :] R_test = np.asfortranarray(Y - np.dot(X, Beta))
    # cdef double[:, :] ZZtop_test = np.dot(R_test, R_test.T) / n_tasks

    for it in range(max_iter):
        if it % f_Sigma_update == 0:
            # R = Y - np.dot(X, Beta):
            tmpint = n_samples * n_tasks
            dcopy(&tmpint, &Y[0, 0], &inc, &R[0, 0], &inc)

            for j in range(n_features):
                for t in range(n_tasks):
                    if Beta[j, t] != 0.:
                        dger(&n_samples, &n_tasks, &minus_one, &X[0, j], &inc,
                            &Beta[j, 0], &inc, &R[0, 0], &n_samples)
                        break

            # R_test = np.asfortranarray(Y - np.dot(X, Beta))
            # np.testing.assert_allclose(R, R_test)

            # ZZtop = np.dot(R, R.T) / n_tasks:
            dgemm(&trans_id, &trans_t, &n_samples, &n_samples, &n_tasks,
                   &inv_n_tasks, &R[0, 0],
                   &n_samples, &R[0, 0], &n_samples, &zero,
                   &ZZtop[0, 0], &n_samples)
            # np.testing.assert_allclose(ZZtop, np.dot(R_test, R_test.T) / n_tasks)

            # U = ZZtop.copy():
            tmpint = n_samples * n_samples
            dcopy(&tmpint, &ZZtop[0, 0], &inc, &U[0, 0], &inc)

            my_eigh(U, lambdas, n_samples, WORK)
            # correct numerical errors:
            for i in range(n_samples):
                if lambdas[i] < 0.:
                    lambdas[i] = 0.

            for i in range(n_samples):
                mus[i] = fmax(sqrt(lambdas[i]), sigma_min)
                invmus[i] = 1. / mus[i]
            # print(np.array(lambdas))
            # print(np.array(mus))

            # Sigma_inv =  U * np.diag(1. / mus) * U^_top:
            tmpint = n_samples * n_samples
            dcopy(&tmpint, &zeros_like_Sigma[0, 0], &inc, &Sigma_inv[0, 0], &inc)
            for i in range(n_samples):
                dger(&n_samples, &n_samples, &invmus[i], &U[0, i], &inc, &U[0, i],
                     &inc, &Sigma_inv[0, 0], &n_samples)

            # np.testing.assert_allclose(Sigma_inv,
                                    #    np.dot(U, np.dot(np.diag(invmus), U.T)))

            # Sigma_inv_R = np.dot(Sigma_inv, R):
            dgemm(&trans_id, &trans_id, &n_samples, &n_tasks, &n_samples,
                  &one, &Sigma_inv[0, 0], &n_samples, &R[0, 0], &n_samples,
                  &zero, &Sigma_inv_R[0, 0], &n_samples)

            # np.testing.assert_allclose(Sigma_inv_R, np.dot(Sigma_inv, R))

            dgemm(&trans_id, &trans_id, &n_samples, &n_features, &n_samples,
                  &one, &Sigma_inv[0, 0], &n_samples, &X[0, 0], &n_samples,
                  &zero, &Sigma_inv_X[0, 0], &n_samples)
            # np.testing.assert_allclose(Sigma_inv_X, np.dot(Sigma_inv, X))


            for j in range(n_features):
                L_const[j] = ddot(&n_samples, &X[0, j], &inc,
                                  &Sigma_inv_X[0, j], &inc)

            # print(np.array(L_const))
            # np.testing.assert_allclose(np.array(L_const),
                                    #    np.dot(X.flat, Sigma_inv_X.flat, axis=0))

            tmpint = n_samples * n_tasks
            p_obj = ddot(&tmpint, &R[0, 0], &inc, &Sigma_inv_R[0, 0], &inc)
            p_obj /= 2. * n_samples * n_tasks
            p_obj += dasum(&n_samples, &mus[0], &inc) / (2 * n_samples)
            for j in range(n_features):
                p_obj += dnrm2(&n_tasks, &Beta[j, 0], &inc) * alpha


            # Theta = Sigma_inv_R / (n_tasks * n_samples * alpha):
            dcopy(&tmpint, &Sigma_inv_R[0, 0], &inc, &Theta[0, 0], &inc)
            tmpdouble = 1. / (n_tasks * n_samples * alpha)
            dscal(&tmpint, &tmpdouble, &Theta[0, 0], &inc)

            dual_norm_XtTheta = 0.
            for j in range(n_features):
                for t in range(n_tasks):
                    XjTheta[t] = ddot(&n_samples, &X[0, j], &inc,
                                      &Theta[0, t], &inc)
                dual_norm_XtTheta = fmax(dual_norm_XtTheta,
                    dnrm2(&n_tasks, &XjTheta[0], &inc))

            # THM: norm(Sigma_inv_R, ord=2) = sqrt(n_tasks) * 1 / max(1, sigma_min / sqrt(lambda[-1]))
            if sqrt(lambdas[-1]) > sigma_min:
                norm2Theta = sqrt(n_tasks)
            else:
                norm2Theta = sqrt(lambdas[-1]) / sigma_min

            norm2Theta /= (n_tasks * n_samples * alpha)

            scal = fmax(dual_norm_XtTheta,
                        alpha * n_samples * sqrt(n_tasks) * norm2Theta)

            if scal > 1.:
                tmpdouble = 1. / scal
                tmpint = n_tasks * n_samples
                dscal(&tmpint, &tmpdouble, &Theta[0, 0], &inc)
            # np.testing.assert_allclose(Theta_test, np.array(Theta))

            d_obj = alpha * ddot(&tmpint, &Y[0, 0], &inc, &Theta[0, 0], &inc)
            d_obj += sigma_min * (0.5 - n_samples *
                     n_tasks * alpha ** 2 * dnrm2(&tmpint, &Theta[0, 0], &inc) ** 2 / 2.)

            if verbose:
                iter_string = "Iteration %d" % it
                print("%s, p_obj=%.8f" % (iter_string, p_obj))
                print("%s  d_obj=%.8f" % (" " * len(iter_string), d_obj))

            gap = p_obj - d_obj
            if gap < tol:
                print("Early exit")
                break

            # np.testing.assert_allclose(np.array(Sigma_inv_R),
            #     np.dot(Sigma_inv, Y - np.dot(X, Beta)))

        for j in range(n_features):
            # WARNING this is probably error prone
            for t in range(n_tasks):
                if Beta[j, t] != 0.:
                    dger(&n_samples, &n_tasks, &one, &Sigma_inv_X[0, j],
                         &inc, &Beta[j, 0], &inc, &Sigma_inv_R[0, 0],
                         &n_samples)
                    break
            for t in range(n_tasks):
                Beta[j, t] = ddot(&n_samples, &X[0, j], &inc,
                                  &Sigma_inv_R[0, t], &inc) / L_const[j]
            # np.testing.assert_allclose(np.array(Beta[j, :]),
                                    #    np.dot(X[:, j], Sigma_inv_R) / L_const[j])
            tmpdouble = alpha * n_samples * n_tasks / L_const[j]

            # in place soft thresholding:
            BST(&Beta[j, 0], n_tasks, tmpdouble, &zeros_like_Beta_g[0, 0],
                n_orient)

            # WARNING this is probably error prone
            for t in range(n_tasks):
                if Beta[j, t] != 0.:
                    dger(&n_samples, &n_tasks, &minus_one,
                         &Sigma_inv_X[0, j],
                         &inc, &Beta[j, 0], &inc, &Sigma_inv_R[0, 0],
                         &n_samples)
                    break

            # np.testing.assert_allclose(np.array(Sigma_inv_R),
            #     np.dot(Sigma_inv, Y - np.dot(X, Beta)))

    return np.array(Beta), np.array(Sigma_inv)


@cython.cdivision(True)
@cython.boundscheck(False)
cpdef multitask_blockhomo_solver(double[::1, :] X, double[::1, :] Y, double alpha,
                            double[:, ::1] Beta_init,
                            int[:] block_indices,
                            double tol=1e-4,
                            long max_iter=10**5,
                            int f_Sigma_update=10,
                            double sigma_min=1e-3, verbose=1,
                            int n_orient=1):
    """Solve the optimization problem of the multi-task BLock Homoscedastic
    Smoothed Concomitant Lasso with alternate minimization.
    We minimise in Beta and Sigma:
    (1/(2 n_samples n_tasks) trace((Y - X Beta).T Sigma^-1 (Y - X Beta)) +
    trace(Sigma) / (2 n_samples) + alpha sum(norm(Beta, ord=2, axis=1))

    with the constraints that Sigma is diagonal, constant over blocks.

    Parameters:
    -----------
        X : Fortran ndarray, shape (n_samples, n_features)
            Training data.
        Y : Fortran ndarray, shape (n_samples, n_tasks)
            Target values.
        alpha : float
            Regularization parameter.
        Beta_init : ndarray, shape (n_features, n_tasks)
            Initialization of the regression coefficients.
        block_indices : ndarray, shape (n_blocks + 1,)
            The k-th block starts at sample block_indices[k] and stop at
            sample block_indices[k + 1] - 1.
        tol : float
            The solver stops when reaching a duality gap lower than tol.
        max_iter : int
            If the duality gap does go below tol after max_iter epochs of
            coordinate descent, the solver stops.
        f_Sigma_update : int
            Number of coordinate descent epochs between updates of noise levels.
        sigma_min : float
            Lower constraint on the noise levels per block.
        verbose : {0, 1}
            Level of verbosity.
        n_orient : {1, 3}
            Use 1 except when dealing with M/EEG applications, in which case
            1 is for fixed orientation and 3 for free orientation.

    Returns:
    --------
        Beta : ndarray, shape (n_features, n_tasks)
            Estimated regression coefficients after optimization.
        sigma_invs : ndarray, shape (n_blocks)
            Inverses of estimated noise levels per block after optimization.
    """
    cdef int n_samples = X.shape[0]
    cdef int n_features = X.shape[1]
    cdef int n_tasks = Y.shape[1]
    cdef int n_groups = n_features // n_orient
    cdef int n_blocks = block_indices.shape[0] - 1
    cdef int[:] block_sizes = np.diff(block_indices)

    cdef double[:] sigmas = np.empty(n_blocks)
    cdef double[:] sigmas_inv = np.empty(n_blocks)

    cdef int it  # iterations
    cdef int k  # blocks
    cdef int i  # samples
    cdef int j  # features
    cdef int g  # groups of features. if n_orient == 1, g and j are the same.
    cdef int o  # orientations
    cdef int t  # tasks
    cdef int inc = 1
    cdef double zero = 0.
    cdef double one = 1.
    cdef double minus_one = - 1.
    cdef int tmpint
    cdef double tmpdouble
    cdef double dual_norm_XtTheta
    cdef double[:] norms_Theta_block = np.empty(n_blocks) # norm of block of Theta


    cdef double[:] L_const = np.empty(n_groups)
    cdef double[:] L_const_test = L_const.copy()
    cdef double[:, ::1] zeros_like_Beta_g = np.zeros([n_orient, n_tasks])
    # TODO E D as lists ?
    cdef double[::1, :] R = np.empty([n_samples, n_tasks], order='F')

    tmpint = n_features * n_tasks
    cdef double[:, :] Beta = np.empty([n_features, n_tasks])
    dcopy(&tmpint, &Beta_init[0, 0], &inc, &Beta[0, 0], &inc)

    cdef double[::1, :] Sigma_inv_X = np.empty([n_samples, n_features], order='F')
    cdef double[::1, :] Sigma_inv_R = np.empty([n_samples, n_tasks], order='F')
    cdef double[::1, :] Theta = np.zeros([n_samples, n_tasks], order='F')
    cdef double[:, ::1] XgTheta = np.zeros([n_orient, n_tasks])
    cdef double[:, ::1] diff_Beta_g = np.zeros([n_orient, n_tasks])

    # to compute spectral norms when n_orient != 1:
    cdef double[::1, :] XjSigma_invXj = np.zeros([n_orient, n_orient], order='F')
    cdef double[:] L_const_eigvals = np.zeros(n_orient)
    cdef double[:] L_const_WORK  = np.empty(3 * n_orient - 1)

    cdef double p_obj
    cdef double d_obj
    cdef double gap
    cdef double scal

    # cdef double[::1, :] R_test = np.asfortranarray(Y - np.dot(X, Beta))

    for it in range(max_iter):
        if it % f_Sigma_update == 0:
            # R = Y - np.dot(X, Beta)
            tmpint = n_samples * n_tasks
            dcopy(&tmpint, &Y[0, 0], &inc, &R[0, 0], &inc)

            for j in range(n_features):
                for t in range(n_tasks):
                    if Beta[j, t] != 0.:
                        dger(&n_samples, &n_tasks, &minus_one, &X[0, j], &inc,
                            &Beta[j, 0], &inc, &R[0, 0], &n_samples)
                        break

            # R_test = np.asfortranarray(Y - np.dot(X, Beta))
            # np.testing.assert_allclose(R, R_test, err_msg="R")

            # after this sigmas will contain the squared residuals per block:
            for k in range(n_blocks):
                sigmas[k] = 0.

                for t in range(n_tasks):
                    sigmas[k] += dnrm2(&block_sizes[k], &R[block_indices[k], t],
                                       &inc) ** 2

                sigmas[k] = sqrt(sigmas[k] / (n_tasks * block_sizes[k]))

                if sigmas[k] < sigma_min:
                    sigmas[k] = sigma_min
                sigmas_inv[k] = 1. / sigmas[k]

            # compute Sigma_inv * X
            tmpint = n_samples * n_features
            dcopy(&tmpint, &X[0, 0], &inc, &Sigma_inv_X[0, 0], &inc)
            # cannot operate directly on lines of X and R because they're Fortran
            for j in range(n_features):
                for k in range(n_blocks):
                    dscal(&block_sizes[k], &sigmas_inv[k],
                          &Sigma_inv_X[block_indices[k], j], &inc)

            # np.testing.assert_allclose(np.array(Sigma_inv_X),
            #                            np.dot(np.diag(np.repeat(sigmas_inv, block_sizes)), X), err_msg="Sigma inv X")

            # compute Sigma_inv * R
            tmpint = n_samples * n_tasks
            dcopy(&tmpint, &R[0, 0], &inc, &Sigma_inv_R[0, 0], &inc)
            for t in range(n_tasks):
                for k in range(n_blocks):
                      dscal(&block_sizes[k], &sigmas_inv[k],
                            &Sigma_inv_R[block_indices[k], t], &inc)

            # np.testing.assert_allclose(np.array(Sigma_inv_R),
            #                            np.dot(np.diag(np.repeat(sigmas_inv, block_sizes)), R), err_msg="Sigma inv R")


            for g in range(n_groups):
                if n_orient == 1:
                    L_const[g] = ddot(&n_samples, &X[0, g], &inc,
                                      &Sigma_inv_X[0, g], &inc)
                else:
                    L_const[g] = spectral_norm_group(X, Sigma_inv_X, g * n_orient,
                                               XjSigma_invXj, L_const_eigvals,
                                               L_const_WORK, n_orient)
            #
            for g in range(n_groups):
                group = slice(g * n_orient, (g + 1) * n_orient)
                L_const_test[g] = np.linalg.norm(np.dot(np.array(X).T[group], np.array(Sigma_inv_X)[:, group]), ord=2)

            np.testing.assert_allclose(np.array(L_const_test), np.array(L_const))


            # print(np.array(L_const))
            # np.testing.assert_allclose(np.array(L_const),
            #                            np.sum(np.array(X) * np.array(Sigma_inv_X), axis=0), err_msg="L const")

            tmpint = n_samples * n_tasks
            p_obj = ddot(&tmpint, &R[0, 0], &inc, &Sigma_inv_R[0, 0], &inc)
            p_obj /= 2. * n_samples * n_tasks
            for k in range(n_blocks):
                p_obj += block_sizes[k] * sigmas[k] / (2 * n_samples)

            tmpint = n_tasks * n_orient
            for g in range(n_groups):
                p_obj += dnrm2(&tmpint, &Beta[g * n_orient, 0], &inc) * alpha

            tmpint = n_samples * n_tasks

            # # Theta = Sigma_inv_R / (n_tasks * n_samples * alpha)
            dcopy(&tmpint, &Sigma_inv_R[0, 0], &inc, &Theta[0, 0], &inc)
            tmpdouble = 1. / (n_tasks * n_samples * alpha)
            dscal(&tmpint, &tmpdouble, &Theta[0, 0], &inc)

            dual_norm_XtTheta = 0.
            tmpint = n_tasks * n_orient  # number of elements in XgTheta
            for g in range(n_groups):
                for t in range(n_tasks):
                    for o in range(n_orient):
                        XgTheta[o, t] = \
                            ddot(&n_samples, &X[0, g * n_orient + o], &inc,
                                 &Theta[0, t], &inc)
                dual_norm_XtTheta = fmax(dual_norm_XtTheta,
                                         dnrm2(&tmpint, &XgTheta[0, 0], &inc))


            for k in range(n_blocks):
                norms_Theta_block[k] = 0.

                for t in range(n_tasks):
                    norms_Theta_block[k] += dnrm2(&block_sizes[k],
                                              &Theta[block_indices[k], t],
                                              &inc) ** 2

                norms_Theta_block[k] = sqrt(norms_Theta_block[k])


            # dual point will be Theta /scal
            scal = 0.
            for k in range(n_blocks):
                scal = fmax(scal, fmax(1, norms_Theta_block[k] * \
                    sqrt(n_tasks / block_sizes[k]) * n_samples * alpha))

            scal = fmax(dual_norm_XtTheta, scal)

            if scal > 1.:
                tmpdouble = 1. / scal
                tmpint = n_tasks * n_samples
                dscal(&tmpint, &tmpdouble, &Theta[0, 0], &inc)

            tmpint = n_tasks * n_samples
            d_obj = alpha * ddot(&tmpint, &Y[0, 0], &inc, &Theta[0, 0], &inc)
            for k in range(n_blocks):
                d_obj += sigma_min * (block_sizes[k] - (n_samples * sqrt(n_tasks) *
                         alpha * norms_Theta_block[k]) ** 2) / (2 * n_samples)


            if verbose:
                iter_string = "Iteration %d" % it
                print("%s, p_obj=%.8f" % (iter_string, p_obj))
                print("%s  d_obj=%.8f" % (" " * len(iter_string), d_obj))

            gap = p_obj - d_obj
            if gap < tol:
                print("Early exit, gap: %s" % format(gap))
                break

            # np.testing.assert_allclose(np.array(Sigma_inv_R),
            #     np.dot(Sigma_inv, Y - np.dot(X, Beta)))

        tmpint = n_orient * n_tasks
        for g in range(n_groups):
            dcopy(&tmpint, &Beta[g * n_orient, 0], &inc,
                  &diff_Beta_g[0, 0], &inc)

            for o in range(n_orient):
                for t in range(n_tasks):
                    Beta[g * n_orient + o, t] += ddot(&n_samples, &X[0, g * n_orient + o], &inc,
                                      &Sigma_inv_R[0, t], &inc) / L_const[g]

            # in place soft thresholding
            tmpdouble = alpha * n_samples * n_tasks / L_const[g]
            BST(&Beta[g * n_orient, 0], n_tasks, tmpdouble,
                &zeros_like_Beta_g[0, 0], n_orient)

            # np.testing.assert_allclose(
            #     max(0, 1 - tmpdouble / np.linalg.norm(test, ord='fro')) * test,
            #     np.array(np.array(Beta)[g * n_orient: (g+1) * n_orient]))

            # diff_Beta_g -= new value of Beta g
            daxpy(&tmpint, &minus_one, &Beta[g * n_orient, 0], &inc,
                  &diff_Beta_g[0, 0], &inc)

            # keep residuals up to date:
            # (do it only if update is non-zero)
            for o in range(n_orient):
                for t in range(n_tasks):
                    if diff_Beta_g[o, t] != 0.:
                        dger(&n_samples, &n_tasks, &one,
                             &Sigma_inv_X[0, g * n_orient + o],
                             &inc, &diff_Beta_g[o, 0], &inc, &Sigma_inv_R[0, 0],
                             &n_samples)
                        break

    return np.array(Beta), np.array(sigmas_inv)


@cython.cdivision(True)
@cython.boundscheck(False)
cpdef multitask_blockhomo_solver_clever(double[::1, :] X, double[::1, :] Y,
                            double alpha,
                            double[:, ::1] Beta_init,
                            int[:] block_indices,
                            double tol=1e-4,
                            long max_iter=10**5,
                            int f_gap=10,
                            double sigma_min=1e-3, verbose=1,
                            int n_orient=1):
    """Solve the optimization problem of the multi-task Block Homoscedastic
    Smoothed Concomitant Lasso with alternate minimization.
    We minimise in Beta and Sigma:
    (1/(2 n_samples n_tasks) trace((Y - X Beta).T Sigma^-1 (Y - X Beta)) +
    trace(Sigma) / (2 n_samples) + alpha sum(norm(Beta, ord=2, axis=1))

    with the constraints that Sigma is diagonal, constant over blocks.

    Parameters:
    -----------
        X : Fortran ndarray, shape (n_samples, n_features)
            Training data.
        Y : Fortran ndarray, shape (n_samples, n_tasks)
            Target values.
        alpha : float
            Regularization parameter.
        Beta_init : ndarray, shape (n_features, n_tasks)
            Initialization of the regression coefficients.
        block_indices : ndarray, shape (n_blocks + 1,)
            The k-th block starts at sample block_indices[k] and stops at
            sample block_indices[k + 1] - 1.
        tol : float
            The solver stops when reaching a duality gap lower than tol.
        max_iter : int
            If the duality gap does not go below tol after max_iter epochs of
            coordinate descent, the solver stops.
        sigma_min : float
            Lower constraint on the noise levels per block.
        verbose : {0, 1}
            Level of verbosity.
        n_orient : {1, 3}
            Use 1 except when dealing with M/EEG applications, in which case
            1 is for fixed orientation and 3 for free orientation.

    Returns:
    --------
        Beta : ndarray, shape (n_features, n_tasks)
            Estimated regression coefficients after optimization.
        sigma_invs : ndarray, shape (n_blocks)
            Inverses of estimated noise levels per block after optimization.
    """
    cdef int n_samples = X.shape[0]
    cdef int n_features = X.shape[1]
    cdef int n_tasks = Y.shape[1]
    cdef int n_groups = n_features // n_orient
    cdef int n_blocks = block_indices.shape[0] - 1
    cdef int[:] block_sizes = np.diff(block_indices)

    cdef double[:] sigmas = np.empty(n_blocks)
    cdef double[:] sigmas_inv = np.empty(n_blocks)

    cdef int it  # iterations
    cdef int k  # blocks
    cdef int i  # samples
    cdef int j  # features
    cdef int g  # groups of features. if n_orient == 1, g and j are the same.
    cdef int o  # orientations
    cdef int t  # tasks
    cdef int inc = 1
    cdef double zero = 0.
    cdef double one = 1.
    cdef double minus_one = - 1.
    cdef int tmpint
    cdef double tmpdouble
    cdef double dual_norm_XtTheta
    cdef double[:] norms_Theta_block = np.empty(n_blocks) # norm of block of Theta


    cdef double[:] L_const = np.empty(n_groups)
    cdef double[:] L_const_test = L_const.copy()
    cdef double[:, ::1] zeros_like_Beta_g = np.zeros([n_orient, n_tasks])
    # TODO E D as lists ?
    cdef double[::1, :] R = np.empty([n_samples, n_tasks], order='F')

    tmpint = n_features * n_tasks
    cdef double[:, :] Beta = np.empty([n_features, n_tasks])
    dcopy(&tmpint, &Beta_init[0, 0], &inc, &Beta[0, 0], &inc)

    cdef double[::1, :] Theta = np.zeros([n_samples, n_tasks], order='F')
    cdef double[:, ::1] XgTheta = np.zeros([n_orient, n_tasks])
    cdef double[:, ::1] diff_Beta_g = np.zeros([n_orient, n_tasks])

    # to compute spectral norms when n_orient != 1:
    cdef double[::1, :] XjSigma_invXj = np.zeros([n_orient, n_orient], order='F')
    cdef double[:] L_const_eigvals = np.zeros(n_orient)
    cdef double[:] L_const_WORK  = np.empty(3 * n_orient - 1)

    cdef double p_obj
    cdef double d_obj
    cdef double gap
    cdef double scal

    # cdef double[::1, :] R_test = np.asfortranarray(Y - np.dot(X, Beta))
    # residuals initialization:
    # R = Y.copy()
    tmpint = n_samples * n_tasks
    dcopy(&tmpint, &Y[0, 0], &inc, &R[0, 0], &inc)
    # R -= np.dot(X, Beta)
    for j in range(n_features):
        for t in range(n_tasks):
            if Beta[j, t] != 0.:
                dger(&n_samples, &n_tasks, &minus_one, &X[0, j],
                     &inc,
                    &Beta[j, 0], &inc, &R[0, 0], &n_samples)
                break

    for it in range(max_iter):
        if True:
            for k in range(n_blocks):
                sigmas[k] = 0.

                for t in range(n_tasks):
                    sigmas[k] += dnrm2(
                        &block_sizes[k],  &R[block_indices[k], t], &inc) ** 2

                sigmas[k] = sqrt(sigmas[k] / (n_tasks * block_sizes[k]))

                if sigmas[k] < sigma_min:
                    sigmas[k] = sigma_min
                sigmas_inv[k] = 1. / sigmas[k]

            for g in range(n_groups):
                L_const[g] = 0.
                if n_orient == 1:
                    for k in range(n_blocks):
                        L_const[g] += dnrm2(&block_sizes[k],
                                            &X[block_indices[k], g],
                                            &inc) ** 2 * sigmas_inv[k]
                else:
                    L_const[g] = spectral_norm_group_clever(
                        X, sigmas_inv, block_indices, block_sizes,
                        g * n_orient, XjSigma_invXj, L_const_eigvals,
                        L_const_WORK, n_orient)


        if it % f_gap == 0:
            p_obj = 0.
            # quadratic data-fitting term:
            for t in range(n_tasks):
                for k in range(n_blocks):
                    p_obj += dnrm2(&block_sizes[k], &R[block_indices[k], t],
                                   &inc) ** 2 * sigmas_inv[k]
            p_obj /= 2. * n_samples * n_tasks
            # sigma penalization:
            for k in range(n_blocks):
                p_obj += block_sizes[k] * sigmas[k] / (2 * n_samples)
            # beta penalization:
            tmpint = n_tasks * n_orient
            for g in range(n_groups):
                p_obj += dnrm2(&tmpint, &Beta[g * n_orient, 0], &inc) * alpha

            # Theta = Sigma^-1 * R / (n_tasks * n_samples * alpha)
            tmpint = n_samples * n_tasks
            dcopy(&tmpint, &R[0, 0], &inc, &Theta[0, 0], &inc)
            for t in range(n_tasks):
                for k in range(n_blocks):
                    tmpdouble = 1. / (n_tasks * n_samples * alpha * sigmas[k])
                    dscal(&block_sizes[k], &tmpdouble,
                          &Theta[block_indices[k], t], &inc)

            dual_norm_XtTheta = 0.
            tmpint = n_tasks * n_orient  # number of elements in XgTheta
            for g in range(n_groups):
                for t in range(n_tasks):
                    for o in range(n_orient):
                        XgTheta[o, t] = \
                            ddot(&n_samples, &X[0, g * n_orient + o], &inc,
                                 &Theta[0, t], &inc)
                dual_norm_XtTheta = fmax(dual_norm_XtTheta,
                                         dnrm2(&tmpint, &XgTheta[0, 0], &inc))

            for k in range(n_blocks):
                norms_Theta_block[k] = 0.

                for t in range(n_tasks):
                    norms_Theta_block[k] += dnrm2(&block_sizes[k],
                                              &Theta[block_indices[k], t],
                                              &inc) ** 2

                norms_Theta_block[k] = sqrt(norms_Theta_block[k])

            # dual point will be Theta /scal
            scal = 0.
            for k in range(n_blocks):
                scal = fmax(scal, fmax(1, norms_Theta_block[k] * \
                    sqrt(n_tasks / block_sizes[k]) * n_samples * alpha))

            scal = fmax(dual_norm_XtTheta, scal)

            if scal > 1.:
                tmpdouble = 1. / scal
                tmpint = n_tasks * n_samples
                dscal(&tmpint, &tmpdouble, &Theta[0, 0], &inc)

            tmpint = n_tasks * n_samples
            d_obj = alpha * ddot(&tmpint, &Y[0, 0], &inc, &Theta[0, 0], &inc)
            for k in range(n_blocks):
                d_obj += sigma_min * (block_sizes[k] - (n_samples * sqrt(n_tasks) *
                         alpha * norms_Theta_block[k]) ** 2) / (2 * n_samples)

            if verbose:
                iter_string = "Iteration %d" % it
                print("%s, p_obj=%.8f" % (iter_string, p_obj))
                print("%s  d_obj=%.8f" % (" " * len(iter_string), d_obj))

            gap = p_obj - d_obj
            if gap < tol:
                print("Early exit, gap: %s" % format(gap))
                break

            # np.testing.assert_allclose(np.array(Sigma_inv_R),
            #     np.dot(Sigma_inv, Y - np.dot(X, Beta)))

        tmpint = n_orient * n_tasks
        for g in range(n_groups):
            dcopy(&tmpint, &Beta[g * n_orient, 0], &inc,
                  &diff_Beta_g[0, 0], &inc)

            for o in range(n_orient):
                for t in range(n_tasks):
                    for k in range(n_blocks):
                        Beta[g * n_orient + o, t] += ddot(
                            &block_sizes[k], &X[block_indices[k], g * n_orient + o],
                            &inc, &R[block_indices[k], t],
                            &inc) / L_const[g] * sigmas_inv[k]

            # in place soft thresholding
            tmpdouble = alpha * n_samples * n_tasks / L_const[g]
            BST(&Beta[g * n_orient, 0], n_tasks, tmpdouble,
                &zeros_like_Beta_g[0, 0], n_orient)


            # diff_Beta_g -= new value of Beta g
            daxpy(&tmpint, &minus_one, &Beta[g * n_orient, 0], &inc,
                  &diff_Beta_g[0, 0], &inc)

            # keep residuals up to date:
            # (do it only if update is non-zero)
            for o in range(n_orient):
                for t in range(n_tasks):
                    if diff_Beta_g[o, t] != 0.:
                        dger(&n_samples, &n_tasks, &one,
                             &X[0, g * n_orient + o],
                             &inc, &diff_Beta_g[o, 0], &inc, &R[0, 0],
                             &n_samples)
                        break


    return np.array(Beta), np.array(sigmas_inv)
