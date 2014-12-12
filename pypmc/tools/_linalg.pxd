"Linear algebra functions, cython header"

cimport numpy as _np

cpdef double bilinear_sym(double[:,:] matrix, double[:] vector)
cpdef object chol_inv_det(m)