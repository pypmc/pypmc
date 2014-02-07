'''Functions to avoid singularities

'''

import numpy as _np

def regularize(x):
    '''Replace zeros by smallest positive float.

    :param x:

        Numpy-array

    Return regularized ``x``.

    '''
    x[_np.where(x == 0)] = _np.finfo('d').tiny
    return x
