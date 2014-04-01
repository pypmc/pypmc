cimport numpy as _np

cpdef double logsumexp(_np.ndarray[double, ndim=1] a, _np.ndarray[double, ndim=1] weights)
cpdef _np.ndarray[double, ndim=1] logsumexp2D(_np.ndarray[double, ndim=2] a, _np.ndarray[double, ndim=1] weights)
