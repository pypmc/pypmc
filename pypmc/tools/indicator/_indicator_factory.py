"""Collect generators of typical indicator functions."""

import numpy as _np

def ball(center, radius=1., bdy=True):
    '''Returns the indicator function of a ball.

    :param center:

        A vector-like numpy array, defining the center of the ball.\n
        len(center) fixes the dimension.

    :param radius:

        Float or int, the radius of the ball

    :param bdy:

        Bool, When ``x`` is at the ball's boundary then
        ``ball_indicator(x)`` returns ``True`` if and only if
        ``bdy=True``.

    '''
    center = _np.array(center) # copy input parameter
    dim = len(center)

    if bdy:
        def ball_indicator(x):
            if len(x) != dim:
                raise ValueError('input has wrong dimension (%i instead of %i)' % (len(x), dim))
            if _np.linalg.norm(x - center) <= radius:
                return True
            return False
    else:
        def ball_indicator(x):
            if len(x) != dim:
                raise ValueError('input has wrong dimension (%i instead of %i)' % (len(x), dim))
            if _np.linalg.norm(x - center) < radius:
                return True
            return False

    # write docstring for ball_indicator
    ball_indicator.__doc__  = 'automatically generated ball indicator function:'
    ball_indicator.__doc__ += '\ncenter = ' + repr(center)[6:-1]
    ball_indicator.__doc__ += '\nradius = ' + str(radius)
    ball_indicator.__doc__ += '\nbdy    = ' + str(bdy)

    return ball_indicator

def hyperrectangle(lower, upper, bdy=True):
    '''Returns the indicator function of a hyperrectangle.

    :param lower:

        Vector-like numpy array, defining the lower boundary of the hyperrectangle.\n
        len(lower) fixes the dimension.

    :param upper:

        Vector-like numpy array, defining the upper boundary of the hyperrectangle.\n

    :param bdy:

        Bool. When ``x`` is at the hyperrectangles's boundary then
        ``hr_indicator(x)`` returns ``True`` if and only if ``bdy=True``.

    '''
    # copy input
    lower = _np.array(lower)
    upper = _np.array(upper)
    dim = len(lower)
    if (upper <= lower).any():
        raise ValueError('invalid input; found upper <= lower')

    if bdy:
        def hr_indicator(x):
            if len(x) != dim:
                raise ValueError('input has wrong dimension (%i instead of %i)' % (len(x), dim))
            if (lower <= x).all() and (x <= upper).all():
                return True
            return False
    else:
        def hr_indicator(x):
            if len(x) != dim:
                raise ValueError('input has wrong dimension (%i instead of %i)' % (len(x), dim))
            if (lower < x).all() and (x < upper).all():
                return True
            return False

    # write docstring for ball_indicator
    hr_indicator.__doc__  = 'automatically generated hyperrectangle indicator function:'
    hr_indicator.__doc__ += '\nlower = ' + repr(lower)[6:-1]
    hr_indicator.__doc__ += '\nupper = ' + repr(upper)[6:-1]
    hr_indicator.__doc__ += '\nbdy   = ' + str(bdy)

    return hr_indicator
