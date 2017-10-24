import numpy as np
cimport numpy as np
from scipy.linalg.cython_blas cimport ddot, dcopy, dscal, dgemm, dger, dnrm2, dasum
from scipy.linalg.cython_lapack cimport dsyev
cimport cython
from libc.math cimport sqrt


cdef inline double fmax(double x, double y) nogil:
    return x if x > y else y


@cython.boundscheck(False)
@cython.cdivision(True)
cpdef my_eigh(double[::1, :] A, double[:] eigvals, int n_samples,
              double[:] WORK):
    """Compute the eigen value decomposition of the Hermitian matrix A.
       BOTH A and eigvals are modified inplace: A will store the eigenvectors"""

    cdef char U_char = 'U'
    cdef char V_char = 'V'
    cdef int INFO  # TODO

    # WORK is an array of length LWORK, used as memory space by LAPACK
    cdef int LWORK = 3 * n_samples - 1
    dsyev(&V_char, &U_char, &n_samples, &A[0, 0], &n_samples, &eigvals[0],
          &WORK[0], &LWORK,
          &INFO)


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef BST(double * vect, int vect_len, double level, double[:] zeros_like_vect):
    cdef int inc = 1
    cdef double vect_norm = dnrm2(&vect_len, vect, &inc)
    if vect_norm == 0.:
        dcopy(&vect_len, &zeros_like_vect[0], &inc, vect, &inc)
        return

    cdef double shrink = 1. - level / vect_norm
    if shrink <= 0.:
        dcopy(&vect_len, &zeros_like_vect[0], &inc, vect, &inc)
    else:
        dscal(&vect_len, &shrink, vect, &inc)


@cython.cdivision(True)
@cython.boundscheck(False)
cpdef multitask_generalized_solver(double[::1, :] X, double[::1, :] Y, double alpha,
                            double[:, ::1] Beta_init, double tol=1e-4,
                            long max_iter=10**5, int f_Sigma_update=10,
                            double sigma_min=1e-3, verbose=1):
    cdef int n_samples = X.shape[0]
    cdef int n_features = X.shape[1]
    cdef int n_tasks = Y.shape[1]
    cdef double inv_n_tasks = 1. / n_tasks

    cdef int it
    cdef int i
    cdef int j
    cdef int inc = 1
    cdef double zero = 0.
    cdef double one = 1.
    cdef double minus_one = - 1.
    cdef int tmpint
    cdef double tmpdouble
    cdef double dual_norm_XtTheta

    cdef char trans_id = 'n'
    cdef char trans_t = 't'

    # Lapack magic
    cdef char V_char = 'V'
    cdef char U_char = 'U'
    cdef int LWORK = 3 * n_samples - 1
    cdef double[:] WORK = np.empty(LWORK)

    # matrix we will use to store the eigevectors of ZZtop
    cdef double[::1, :] U = np.empty([n_samples, n_samples], order='F')

    cdef double[:] L_const = np.empty(n_features)
    cdef double[:] zeros_like_Beta_j = np.zeros(n_tasks)
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
                        print(j)
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
            BST(&Beta[j, 0], n_tasks, tmpdouble, zeros_like_Beta_j)

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
                            double sigma_min=1e-3, verbose=1):
    cdef int n_samples = X.shape[0]
    cdef int n_features = X.shape[1]
    cdef int n_tasks = Y.shape[1]
    cdef int n_blocks = block_indices.shape[0] - 1
    cdef int[:] block_sizes = np.diff(block_indices)

    cdef double[:] sigmas = np.empty(n_blocks)
    cdef double[:] sigmas_inv = np.empty(n_blocks)

    cdef int it  # iterations
    cdef int k  # blocks
    cdef int i  # samples
    cdef int j  # features
    cdef int t  # tasks
    cdef int inc = 1
    cdef double zero = 0.
    cdef double one = 1.
    cdef double minus_one = - 1.
    cdef int tmpint
    cdef double tmpdouble
    cdef double dual_norm_XtTheta
    cdef double[:] norms_Theta_block = np.empty(n_blocks) # norm of block of Theta


    cdef double[:] L_const = np.empty(n_features)
    cdef double[:] zeros_like_Beta_j = np.zeros(n_tasks)
    # TODO E D as lists ?
    cdef double[::1, :] R = np.empty([n_samples, n_tasks], order='F')

    tmpint = n_features * n_tasks
    cdef double[:, :] Beta = np.empty([n_features, n_tasks])
    dcopy(&tmpint, &Beta_init[0, 0], &inc, &Beta[0, 0], &inc)

    cdef double[::1, :] Sigma_inv_X = np.empty([n_samples, n_features], order='F')
    cdef double[::1, :] Sigma_inv_R = np.empty([n_samples, n_tasks], order='F')
    cdef double[::1, :] Theta = np.zeros([n_samples, n_tasks], order='F')
    cdef double [:] XjTheta = np.zeros(n_tasks)


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
            for k in range(n_blocks):
                sigmas[k] = sqrt(sigmas[k] / (n_tasks * block_sizes[k]))

            for k in range(n_blocks):
                if sigmas[k] < sigma_min:
                    sigmas[k] = sigma_min
                sigmas_inv[k] = 1. / sigmas[k]

            # compute Sigma_inv * X
            # and Sigma_inv R
            tmpint = n_samples * n_features
            dcopy(&tmpint, &X[0, 0], &inc, &Sigma_inv_X[0, 0], &inc)
            # cannot operate directly on lines of X and R because they're fortran
            for j in range(n_features):
                for k in range(n_blocks):
                    dscal(&block_sizes[k], &sigmas_inv[k],
                          &Sigma_inv_X[block_indices[k], j], &inc)

            # np.testing.assert_allclose(np.array(Sigma_inv_X),
            #                            np.dot(np.diag(np.repeat(sigmas_inv, block_sizes)), X), err_msg="Sigma inv X")


            tmpint = n_samples * n_tasks
            dcopy(&tmpint, &R[0, 0], &inc, &Sigma_inv_R[0, 0], &inc)
            for t in range(n_tasks):
                for k in range(n_blocks):
                      dscal(&block_sizes[k], &sigmas_inv[k],
                            &Sigma_inv_R[block_indices[k], t], &inc)

            # np.testing.assert_allclose(np.array(Sigma_inv_R),
            #                            np.dot(np.diag(np.repeat(sigmas_inv, block_sizes)), R), err_msg="Sigma inv R")


            for j in range(n_features):
                L_const[j] = ddot(&n_samples, &X[0, j], &inc,
                                  &Sigma_inv_X[0, j], &inc)

            # print(np.array(L_const))
            # np.testing.assert_allclose(np.array(L_const),
            #                            np.sum(np.array(X) * np.array(Sigma_inv_X), axis=0), err_msg="L const")

            tmpint = n_samples * n_tasks
            p_obj = ddot(&tmpint, &R[0, 0], &inc, &Sigma_inv_R[0, 0], &inc)
            p_obj /= 2. * n_samples * n_tasks
            for k in range(n_blocks):
                p_obj += block_sizes[k] * sigmas[k] / (2 * n_samples)

            for j in range(n_features):
                p_obj += dnrm2(&n_tasks, &Beta[j, 0], &inc) * alpha


            tmpint = n_samples * n_tasks

            # # Theta = Sigma_inv_R / (n_tasks * n_samples * alpha)
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

            for k in range(n_blocks):
                norms_Theta_block[k] = 0.

                for t in range(n_tasks):
                    norms_Theta_block[k] += dnrm2(&block_sizes[k],
                                              &Theta[block_indices[k], t],
                                              &inc) ** 2
            for k in range(n_blocks):
                norms_Theta_block[k] = sqrt(norms_Theta_block[k])



            # dual point will be Theta /scal
            scal = 0.
            for k in range(n_blocks):
                scal = fmax(scal, fmax(1, norms_Theta_block[k] * np.sqrt(n_tasks) * n_samples * alpha) / sqrt(block_sizes[k]))

            scal = fmax(dual_norm_XtTheta, scal)

            if scal > 1.:
                tmpdouble = 1. / scal
                tmpint = n_tasks * n_samples
                dscal(&tmpint, &tmpdouble, &Theta[0, 0], &inc)

            d_obj = alpha * ddot(&tmpint, &Y[0, 0], &inc, &Theta[0, 0], &inc)
            for k in range(n_blocks):
                d_obj += sigma_min * (block_sizes[k] - (n_samples * np.sqrt(n_tasks) *
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

            # in place soft thresholding
            BST(&Beta[j, 0], n_tasks, tmpdouble, zeros_like_Beta_j)

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

    return np.array(Beta), np.array(sigmas_inv)
