'Functions to avoid singularities'

import numpy as _np
from libc.math cimport exp, log

def regularize(x):
    '''Replace zeros by smallest positive float.

    :param x:

        Numpy-array

    Return regularized ``x``.

    '''
    x[x == 0] = _np.finfo('d').tiny
    return x

cpdef double logsumexp(_np.ndarray[double, ndim=1] a, _np.ndarray[double, ndim=1] weights):
    r'''Replacement for scipy's logsumexp for 1D vectors of double-precision floats.
    Return

    .. math::

           \log \sum_i w_i \exp(a_i)

    computed as

    .. math::

        \max_i a + \log ( \sum_i w_i exp(a_i - \max_i a) ).

    :param a:
        1D double array; The logarithms to be added on the linear scale.

    :param weights:
        1D double array; The weights of the logarithmic terms.

    '''
    assert a is not None
    assert weights is not None

    cdef:
        size_t i
        double max_val = -_np.finfo('d').max
        double res = 0.0

    for i in range(len(a)):
        if a[i] > max_val:
            max_val = a[i]

    for i in range(len(a)):
        res += weights[i] * exp(a[i] - max_val)

    return log(res) + max_val

cpdef _np.ndarray[double, ndim=1] logsumexp2D(_np.ndarray[double, ndim=2] a, _np.ndarray[double, ndim=1] weights):
    r'''Replacement for scipy's logsumexp for 2D array of double-precision
        floats. The one-dimensional :py:func:`.logsumexp` is applied to
        each row.

    '''
    assert a is not None
    assert weights is not None
    assert (weights >= 0.).all(), 'Found negative weight'

    cdef:
        size_t k,n
        double max_val
        _np.ndarray[double, ndim=1] res = _np.zeros(len(a))

    for n in range(len(a)):
        max_val = -_np.finfo('d').max
        for k in range(len(a[0])):
            if a[n,k] > max_val:
                max_val = a[n,k]

        for k in range(len(a[0])):
            res[n] += weights[k] * exp(a[n,k] - max_val)

        res[n] = log(res[n]) + max_val

    return res
