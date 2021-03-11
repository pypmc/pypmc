'''Provide functions to rate the quality of weighted samples.

'''
import numpy as _np

def perp(weights):
    r"""Calculate the normalized perplexity :math:`\mathcal{P}` of samples
    with ``weights`` :math:`\omega_i`. :math:`\mathcal{P}=0` is
    terrible and :math:`\mathcal{P}=1` is perfect.

    .. math::

        \mathcal{P} = exp(H) / N

    where

    .. math::

        H = - \sum_{i=1}^N \bar{\omega}_i log ~ \bar{\omega}_i

    .. math::

        \bar{\omega}_i = \frac{\omega_i}{\sum_i \omega_i}

    :param weights:

        Vector-like array; the samples' weights

    """
    # normalize weights
    w = _np.asarray(weights) / _np.sum(weights)

    # mask zero weights
    w = _np.ma.MaskedArray(w, copy=False, mask=(w == 0))

    # avoid NaN due to log(0) by log(1)=0
    entr = - _np.sum( w * _np.log(w.filled(1.0)))

    return  _np.exp(entr) / len(w)


def ess(weights):
    r"""Calculate the normalized effective sample size :math:`ESS` [LC95]_
    of samples with ``weights`` :math:`\omega_i`.  :math:`ESS=0` is
    terrible and :math:`ESS=1` is perfect.

    .. math::

        ESS = \frac{1}{1+C^2}

    where

    .. math::

        C^2 = \frac{1}{N} \sum_{i=1}^N (N \bar{\omega}_i - 1)^2

    .. math::

        \bar{\omega}_i = \frac{\omega_i}{\sum_i \omega_i}

    :param weights:

        Vector-like array; the samples' weights

    """
    # normalize weights
    w = _np.asarray(weights) / _np.sum(weights)

    # ess
    coeff_var = _np.sum((len(w) * w - 1)**2) / len(w)

    return  1.0 / (1.0 + coeff_var)
