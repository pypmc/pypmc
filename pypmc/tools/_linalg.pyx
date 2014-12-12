"Linear algebra functions"

import numpy as np
import scipy
from scipy.linalg.decomp_cholesky import cholesky
from scipy.linalg.lapack import get_lapack_funcs

from libc.math cimport log

cpdef double bilinear_sym(double[:,:] matrix, double[:] vector):
    '''Compute the bilinear form of a *symmetric* matrix and a vector.

    Return
    .. math::

           x^T \cdot M \cdot x

    where :math:`M=M^T`.

    .. warning::

           No checking is performed on the size of the inputs, or if
           the matrix indeed is symmetric. Expect segfault if
           assumptions are violated.

    '''

    cdef:
        double res = 0.0
        size_t i, j

    for i in range(len(vector)):
        # diagonal contribution
        res += vector[i] * vector[i] * matrix[i,i]
        for j in range(i):
            # off-diagonal elements come twice
            res += 2. * vector[i] * vector[j] * matrix[i,j]

    return res

cpdef object chol_inv_det(m):
    r'''Compute the Cholesky decomposition of a real symmetric matrix :math:`M`
    and use it to compute the inverse and the log of the determinant.

    Compute

    .. math::

            M = L L^T

    Return `L`, :math:`M^{-1}`, :math`\log(\det M)`.

    '''

    cdef:
        size_t i, j, d
        double log_det = 0.0
        double [:,:] inverse_view, lower_view

    m = np.asarray_chkfinite(m)

    if not np.allclose(m, m.T):
        raise np.linalg.LinAlgError('matrix not symmetric:\n' + repr(m))

    # lower triangular decomposition
    lower = True
    l = cholesky(m, lower)
    lower_view = l
    dpotri = get_lapack_funcs('potri', (m,))

    # work around a bug in older versions of scipy
    # that inverts the lower argument
    from pkg_resources import parse_version
    if parse_version(scipy.__version__) < parse_version('0.14'):
        lower = not lower

    # inverse from lower Cholesky
    inverse = dpotri(l, lower)[0]
    inverse_view = inverse

    # symmetrize: copy upper to lower half
    # and compute determinant while we are at it
    d = len(m)
    for i in range(d):
        log_det += log(lower_view[i,i])
        for j in range(i):
            inverse_view[j,i] = inverse_view[i,j]

    # det(M) = det(L L^T) => log det(M) = 2 * log det(L)
    log_det *= 2.0

    if not np.isfinite(log_det):
        raise np.linalg.LinAlgError('Nonpositive eigenvalues lead to invalid determinant ' + repr(log_det))

    return l, inverse, log_det
