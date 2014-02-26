"""Functions to handle mixture densities.

"""
from ..importance_sampling.proposal import MixtureDensity, Gauss
import numpy as _np

def create_gaussian_mixture(means, covs, weights=None):
    """Creates a :py:class:`.MixtureDensity` with gaussian (:py:class:`.Gauss`)
    components. The output can be used for the clustering algorithms.

    :param means:

        Vector-like array; the means of the gaussian mixture

    :param covs:

        3d-matrix-like array; the covariances of the gaussian mixture.
        cov[i] will be interpreted as the i'th covariance matrix.

    :param weights:

        Vector-like array, optional; the component weights. If not
        provided all components will get equal weight.

        .. note::

            The weights will automatically be normalized.

    """
    assert len(means) == len(covs), 'number of means (%i) does not match number of covariances (%i)' %(len(means), len(covs) )
    components = []
    for mean, cov in zip(means, covs):
        components.append(Gauss(mean, cov))
    return MixtureDensity(components, weights)


def recover_gaussian_mixture(mixture):
    """Extracts the means, covariances and component weights from a
    :py:class:`.MixtureDensity`.

    :param mixture:

        :py:class:`.MixtureDensity` with gaussian (:py:class:`.Gauss`) components;
        the mixture to be decomposed.

    """
    weights = _np.array(mixture.weights)

    dim = mixture.dim
    N   = len(weights)

    means = _np.empty((N, dim))
    covs  = _np.empty((N, dim, dim))

    for i, c in enumerate(mixture.components):
        means[i] = c.mu
        covs [i] = c.sigma

    return means, covs, weights
