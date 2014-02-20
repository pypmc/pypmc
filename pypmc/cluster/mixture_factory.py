"""Functions to handle mixture densities.

"""
from ..pmc.proposal import MixtureProposal, GaussianComponent
import numpy as _np

def create_gaussian_mixture(means, covs, weights=None):
    """Creates a :py:class:`pypmc.pmc.proposal.MixtureProposal`
    with gaussian (:py:class:`pypmc.pmc.proposal.GaussianComponent`)
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
        components.append(GaussianComponent(mean, cov))
    return MixtureProposal(components, weights)


def recover_gaussian_mixture(mixture):
    """Extracts the means, covariances and component weights from a
    :py:class:`pypmc.pmc.proposal.MixtureProposal`.

    :param mixture:

        :py:class:`pypmc.pmc.proposal.MixtureProposal` with components of
        :py:class:`pypmc.pmc.proposal.GaussianComponent`; the mixture
        to be decomposed.

    """
    weights = mixture.weights.copy()

    dim = mixture.dim
    N   = len(weights)

    means = _np.empty((N, dim))
    covs  = _np.empty((N, dim, dim))

    for i, c in enumerate(mixture.components):
        means[i] = c.mu
        covs [i] = c.sigma

    return means, covs, weights
