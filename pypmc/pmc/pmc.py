"""Collect Population Monte Carlo

"""

from .proposal import MixtureProposal, GaussianComponent
import numpy as _np
from math import exp as _exp
from copy import deepcopy as _cp
from .._tools._regularize import regularize

def gaussian_pmc(samples, proposal, origin=None, copy=True):
    '''Adapts a ``proposal`` using the (M-)PMC algorithm according to
    [Cap+08]_.

    :param samples:

        Matrix-like array; the samples to be used for the pmc-run. The first
        column is interpreted as (unnormalized) weights.

    :param proposal:

        :py:class:`.MixtureProposal` with :py:class:`.GaussianComponent` s;
        the proposal which proposed the ``samples`` and shall be updated.

    :param origin:

        Vector-like array of integers, optional;
        If not provided, the component which proposed a sample is considered
        as latent variable (unknown). This implements the Rao-Blackwellized
        algorithm.
        If provided, each sample only updates its responsible component.
        This non-Rao-Blackwellized scheme is faster but less accurate.

    :param copy:

        Bool; If True (default), the parameter ``proposal`` remains untouched.
        Otherwise, ``proposal`` is overwritten by the adapted proposal.

    '''
    if copy:
        proposal = _cp(proposal)

    weights = samples[:,0 ]
    samples = samples[:,1:]
    normalized_weights = weights / weights.sum()

    if origin is None:
        rho = _np.empty(( len(samples),len(proposal.components) ))
        for k in range(len(proposal.components)):
            for n, sample in enumerate(samples):
                rho[n, k]  = _exp(proposal.components[k].evaluate(sample)) * proposal.weights[k]
                # avoid division by zero
                rho[n, k] /= _exp(proposal.evaluate(sample)) + _np.finfo('d').tiny
    else:
        rho = _np.zeros(( len(samples),len(proposal.components) ), dtype=bool)
        for i in range(len(proposal.components)):
            rho[_np.where(origin==i),i] = True


    # -------------- update equations according to (14) in [Cap+08] --------------

    # new component weights
    alpha = _np.einsum('n,nk->k', normalized_weights, rho)
    regularize(alpha)
    inv_alpha = 1./alpha

    # new means
    mu = _np.einsum('n,nk,nd->kd', normalized_weights, rho, samples)
    mu = _np.einsum('kd,k->kd', mu, inv_alpha)

    # new covars
    cov = _np.empty(( len(mu),len(samples[0]),len(samples[0]) ))
    for k in range(len(proposal.components)):
        x_minus_mu = samples - mu[k]
        cov[k] = _np.einsum('n,n,ni,nj->ij', normalized_weights, rho[:,k], x_minus_mu, x_minus_mu) * inv_alpha[k]

    # ----------------------------------------------------------------------------

    # apply the updated mixture weights, means and covariances
    for k, component in enumerate(proposal.components):
        proposal.weights[k] = alpha[k]
        # if matrix is not positive definite, the update will fail
        # in that case replug the old values
        old_mu    = component.mu    # do not need to copy because .update creates a new array
        old_sigma = component.sigma # do not need to copy because .update creates a new array
        try:
            component.update(mu[k], cov[k])
        except _np.linalg.LinAlgError:
            component.update(old_mu, old_sigma)

    return proposal
