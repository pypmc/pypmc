"""Collect Population Monte Carlo

"""

from __future__ import division
from .proposal import MixtureProposal, GaussianComponent
import numpy as _np
from math import exp as _exp
from copy import deepcopy as _cp
from ..tools._regularize import regularize

def gaussian_pmc(weighted_samples, proposal, origin=None, rb=True, mincount=0, copy=True, weighted=True):
    '''Adapts a ``proposal`` using the (M-)PMC algorithm according to
    [Cap+08]_.

    :param weighted_samples:

        Matrix-like array; the samples to be used for the pmc-run. The first
        column is interpreted as (unnormalized) weights unless ``weighted``
        is `False`.

    :param proposal:

        :py:class:`.MixtureProposal` with :py:class:`.GaussianComponent` s;
        the proposal which proposed the ``weighted_samples`` and shall be
        updated.

    :param origin:

        Vector-like array of integers, optional; the indices of the responsible
        components for each sample.

    :param rb:

        Bool;
        If True, the component which proposed a sample is considered
        as latent variable (unknown). This implements the Rao-Blackwellized
        algorithm.
        If False, each sample only updates its responsible component. This
        non-Rao-Blackwellized scheme is faster but only an approximation.

    :param mincount:

        Integer; The minimum number of samples a component must have proposed
        in order to not get weight zero. A value of zero (default) disables
        this feature.

        .. important::

            Only possible if ``origin`` is provided.

        .. hint::

            For those components, no calculations are performed which
            results in speed up.

    :param copy:

        Bool; If True (default), the parameter ``proposal`` remains untouched.
        Otherwise, ``proposal`` is overwritten by the adapted proposal.

    :param weighted:

        Bool; If True (default), the first column of ``samples`` is interpreted
        as weights.
        If False, the first column of ``samples`` is interpreted as the first
        coordinate.

    '''
    if weighted:
        weights = weighted_samples[:,0 ]
        samples = weighted_samples[:,1:]
        normalized_weights = weights / weights.sum()
    else:
        samples = weighted_samples

    def calculate_rho_rb():
        # if a component is pruned, the other weights must be renormalized
        need_renormalize = False
        rho = _np.zeros(( len(weighted_samples),len(proposal.components) ))
        for k in range(len(proposal.components)):
            if proposal.weights[k] == 0.:
                # skip unneccessary calculation
                continue
            elif count[k] < mincount:
                proposal.weights[k] = 0.
                need_renormalize = True
                print("Component %i died because of too few (%i) samples." %(k, count[k]))
                continue
            else:
                for n, sample in enumerate(samples):
                    rho[n, k]  = _exp(proposal.components[k].evaluate(sample)) * proposal.weights[k]
                    # + "tiny" --> avoid division by zero
                    rho[n, k] /= _exp(proposal.evaluate(sample)) + _np.finfo('d').tiny
        return rho, need_renormalize

    def calculate_rho_non_rb():
        # if a component is pruned, the other weights must be renormalized
        need_renormalize = False
        rho = _np.zeros(( len(samples),len(proposal.components) ), dtype=bool)
        for k in range(len(proposal.components)):
            if proposal.weights[k] == 0.:
                # skip unneccessary calculation
                continue
            elif count[k] < mincount:
                proposal.weights[k] = 0.
                need_renormalize = True
                print("Component %i died because of too few (%i) samples." %(k, count[k]))
                continue
            else:
                rho[_np.where(origin==k),k] = True
        return rho, need_renormalize

    if copy:
        proposal = _cp(proposal)

    if origin is None:
        if mincount > 0:
            raise ValueError('`mincount` must be 0 if `origin` is not provided!')
        if not rb:
            raise ValueError('`rb` must be True if `origin` is not provided!')
        count = _np.ones(len(proposal.components))
        rho, need_renormalize = calculate_rho_rb()


    else: # if origin is not None
        count = _np.histogram(origin, bins=len(proposal.components), range=(0,len(proposal.components)))[0]
        if rb:
            rho, need_renormalize = calculate_rho_rb()
        else:
            rho, need_renormalize = calculate_rho_non_rb()

    # -------------- update equations according to (14) in [Cap+08] --------------

    # new component weights
    if weighted:
        alpha = _np.einsum('n,nk->k', normalized_weights, rho)
    else:
        alpha = _np.einsum('nk->k', rho) / len(samples)
    inv_alpha = 1./regularize(alpha)

    # new means
    if weighted:
        mu = _np.einsum('n,nk,nd->kd', normalized_weights, rho, samples)
    else:
        mu = _np.einsum('nk,nd->kd', rho, samples) / len(samples)
    mu = _np.einsum('kd,k->kd', mu, inv_alpha)

    # new covars
    cov = _np.empty(( len(mu),len(samples[0]),len(samples[0]) ))
    for k in range(len(proposal.components)):
        if proposal.weights[k] == 0.:
            # skip unneccessary calculation
            continue
        else:
            x_minus_mu = samples - mu[k]
            if weighted:
                cov[k] = _np.einsum('n,n,ni,nj->ij', normalized_weights, rho[:,k], x_minus_mu, x_minus_mu) * inv_alpha[k]
            else:
                cov[k] = _np.einsum('n,ni,nj->ij', rho[:,k], x_minus_mu, x_minus_mu) * inv_alpha[k]
    if not weighted:
        cov /= len(samples)

    # ----------------------------------------------------------------------------

    # apply the updated mixture weights, means and covariances
    for k, component in enumerate(proposal.components):
        if proposal.weights[k] == 0.:
            # skip unneccessary calculation
            continue
        else:
            proposal.weights[k] = alpha[k]
            # if matrix is not positive definite, the update will fail
            # in that case replug the old values and set its weight to zero
            old_mu    = component.mu    # do not need to copy because .update creates a new array
            old_sigma = component.sigma # do not need to copy because .update creates a new array
            try:
                component.update(mu[k], cov[k])
            except _np.linalg.LinAlgError:
                print("Could not update component %i --> weight is set to zero." %k)
                component.update(old_mu, old_sigma)
                proposal.weights[k] = 0.
                need_renormalize = True
    if need_renormalize:
        proposal.normalize()

    return proposal
