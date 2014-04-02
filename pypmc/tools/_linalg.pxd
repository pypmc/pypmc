"Linear algebra functions, cython header"

cimport numpy as _np

cpdef double bilinear_sym(double[:,:] matrix, double[:] vector)
