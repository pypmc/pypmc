"""Collect Population Monte Carlo

"""

from __future__ import division
from ..density.gauss import Gauss
from ..density.mixture import MixtureDensity
import numpy as _np
from math import exp as _exp
from copy import deepcopy as _cp
from ..tools._regularize import regularize

from libc.math cimport exp
cimport numpy as _np

def gaussian_pmc(samples, density, weights=None, latent=None, rb=True, mincount=0, copy=True):
    '''Adapt a mixture ``density`` using the (M-)PMC algorithm according
    to [Cap+08]_.

    :param samples:

        Matrix-like array; the samples to be used for the PMC run.

    :param density:

        :py:class:`.MixtureDensity` with :py:class:`.Gauss` components;
        the density which proposed the ``samples`` and shall be
        updated.

    :param weights:

        Vector-like array of floats; The (unnormalized) importance
        weights. If not given, assume all samples have equal weight.

    :param latent:

        Vector-like array of integers, optional; the latent variables
        (indices) of the generating components for each sample.

    :param rb:

        Bool;
        If True, the component which proposed a sample is considered
        as a latent variable (unknown). This implements the Rao-Blackwellized
        algorithm.
        If False, each sample only updates its responsible component. This
        non-Rao-Blackwellized scheme is faster but only an approximation.

    :param mincount:

        Integer; The minimum number of samples a component has to
        generate in order not to be ignored during updates. A value of
        zero (default) disables this feature. The motivation is that
        components with very small weight generate few samples, so the
        updates become unstable and it is more efficient to simply assign
        weight zero.

        .. important::

            Only possible if ``latent`` is provided.

        .. seealso::

            :py:meth:`.MixtureDensity.prune`

    :param copy:

        Bool; If True (default), the parameter ``density`` remains untouched.
        Otherwise, ``density`` is overwritten by the adapted density.

    '''
    need_renormalize = False

    if copy:
        density = _cp(density)

    def calculate_rho_rb():
        rho = _np.zeros(( len(samples),len(density.components) ))

        cdef:
            size_t n
            int    k
            double tiny = _np.finfo('d').tiny
            double [:] component_weights = density.weights
            double [:] log_denominator = density.multi_evaluate(samples, rho, live_components)
            double [:,:] memview_rho = rho

        for k in live_components:
            for n in range(len(samples)):
                memview_rho[n, k]  = exp(memview_rho[n, k]) * component_weights[k]
                # + "tiny" --> avoid division by zero
                memview_rho[n, k] /= exp(log_denominator[n]) + tiny
        return rho

    def calculate_rho_non_rb():
        rho = _np.zeros(( len(samples),len(density.components) ))
        for k in live_components:
            rho[latent==k,k] = 1.
        return rho

    if weights is not None:
        weights = _np.asarray(weights)
        assert len(weights.shape) == 1, 'Weights must be one-dimensional.'
        assert len(weights) == len(samples), \
            "Number of weights (%s) does not match the number of samples (%s)." % (len(weights), len(samples))
        normalized_weights = weights / weights.sum()

    if latent is None:
        if mincount > 0:
            raise ValueError('`mincount` must be 0 if `latent` is not provided!')
        if not rb:
            raise ValueError('`rb` must be True if `latent` is not provided!')

        # set up list of live_components
        live_components = []
        for k in range(len(density)):
            if density.weights[k] != 0:
                live_components.append(k)

        rho = calculate_rho_rb()


    else: # if latent is not None
        count = _np.histogram(latent, bins=len(density.components), range=(0,len(density.components)))[0]

        # set up list of live_components
        live_components = []
        for k in range(len(density)):
            if density.weights[k] == 0.:
                continue
            live_components.append(k)

        if rb:
            rho = calculate_rho_rb()
        else:
            rho = calculate_rho_non_rb()

        # prune components with less samples than ``mincount`` AFTER rho has been calculated
        for k in live_components:
            if count[k] < mincount:
                live_components.pop(k)
                density.weights[k] = 0.
                # when a component is pruned, the other weights must be renormalized
                need_renormalize = True
                print("Component %i died because of too few (%i) samples." %(k, count[k]))

    # -------------- update equations according to (14) in [Cap+08] --------------

    # allocate memory for covariances (other memory is allocated on demand)
    cov = _np.empty(( len(density.components),len(samples[0]),len(samples[0]) ))
    x_minus_mu = _np.empty((len(samples), len(samples[0])))


    if weights is not None:

        # new component weights
        alpha = _np.einsum('n,nk->k', normalized_weights, rho)
        inv_alpha = 1. / regularize(alpha)

        # new means
        mu = _np.einsum('n,nk,ni->ki', normalized_weights, rho, samples)
        mu = _np.einsum('ki,k->ki', mu, inv_alpha)

        # new covars
        for k in live_components:
            x_minus_mu[:] = samples
            x_minus_mu -= mu[k]
            _np.einsum('n,n,ni,nj->ij', normalized_weights, rho[:,k], x_minus_mu, x_minus_mu, out=cov[k])
            cov[k] *= inv_alpha[k]

    else: # if weights is None

        # new component weights
        alpha = _np.einsum('nk->k', rho) / len(samples)
        inv_alpha = 1. / regularize(alpha)

        # new means
        mu = _np.einsum('nk,ni->ki', rho, samples) / len(samples)
        mu = _np.einsum('ki,k->ki', mu, inv_alpha)

        # new covars
        for k in live_components:
            x_minus_mu[:] = samples
            x_minus_mu -= mu[k]
            _np.einsum('n,ni,nj->ij', rho[:,k], x_minus_mu, x_minus_mu, out=cov[k])
            cov[k] *= inv_alpha[k]

        cov /= len(samples)

    # ----------------------------------------------------------------------------

    # apply the updated mixture weights, means and covariances
    for k in live_components:
        component = density.components[k]
        density.weights[k] = alpha[k]
        # if matrix is not positive definite, the update will fail
        # in that case replug the old values and set its weight to zero
        old_mu    = component.mu    # do not need to copy because .update creates a new array
        old_sigma = component.sigma # do not need to copy because .update creates a new array
        try:
            component.update(mu[k], cov[k])
        except _np.linalg.LinAlgError:
            print("Could not update component %i --> weight is set to zero." %k)
            component.update(old_mu, old_sigma)
            density.weights[k] = 0.
            # when a component is pruned, the other weights must be renormalized
            need_renormalize = True

    if need_renormalize:
        density.normalize()

    return density
