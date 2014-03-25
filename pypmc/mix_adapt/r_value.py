'''Merge Gaussians with a sufficiently small Gelman-Rubin R-value [GR92]_.

'''

from __future__ import division as _div
import numpy as _np

def r_value(means, variances, n, approx=False):
    '''Calculate the Gelman-Rubin R-value (Chapter 2.2 in [GR92]_).
    The R-value can be used to quantify convergence of "Iterative
    Simulations" (e.g. Markov Chains) to their limiting (target)
    distribution. An R-value "close to one" indicates convergence.

    .. note::

        The R-value is defined for univariate distributions only.


    :param means:

        Vector-like array; the mean value estimates

    :param variances:

        Vector-like array; the variance estimates

    :param n:

        Integer; the number of samples used to determine the estimates
        passed via ``means`` and ``variances``

    :approx:

        Bool; If False (default), calculate the R-value as in [GR92]_.
        If True, neglect the uncertainty induced by the sampling process.

    '''
    # use same variable names as in [GR92]
    # means is \bar{x}_i
    # variances is s_i^2

    means     = _np.asarray(means)
    variances = _np.asarray(variances)

    assert len(means.shape) == 1, '``means`` must be vector-like'
    assert len(variances.shape) == 1, '``variances`` must be vector-like'
    assert len(means) == len(variances), \
    'Number of ``means`` (%i) does not match number of ``variances`` (%i)' %( len(means), len(variances) )

    m = len(means)

    x_bar    = _np.average(means)
    B_over_n = ((means - x_bar)**2).sum() / (m - 1)

    W = _np.average(variances)

    # var_estimate is \hat{\sigma}^2
    var_estimate = (n - 1) / n  *  W   +   B_over_n

    if approx:
        return var_estimate / W

    V = var_estimate + B_over_n / m

    # calculate the three terms of \hat{var}(\hat{V}) (equation (4) in [GR92]
    # var_V is \hat{var}(\hat{V})
    tmp_cov_matrix = _np.cov(variances, means)
    # third term
    var_V = _np.cov(variances, means**2)[1,0] - 2. * x_bar * tmp_cov_matrix[1,0]
    var_V *= 2. * (m + 1) * (n - 1) / (m * m * n)
    # second term (merged n in denominator into ``B_over_n``)
    var_V += ((m + 1) / m)**2 * 2. / (m - 1) * B_over_n * B_over_n
    # first term
    var_V += ((n - 1) / n)**2 / m * tmp_cov_matrix[0,0]

    df = 2. * V**2 / var_V

    if df <= 2.:
        return _np.inf

    return V / W * df / (df - 2)
