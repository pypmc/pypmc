"""Collect Population Monte Carlo

"""

from __future__ import division
from ..density.gauss import Gauss
from ..density.student_t import StudentT
from ..density.mixture import MixtureDensity
from ..tools._regularize import regularize
from copy import deepcopy as _cp
from scipy.special import digamma as _psi
from scipy.optimize import brentq as _find_root
import numpy as _np

from pypmc.tools._regularize cimport logsumexp2D
from pypmc.tools._linalg cimport bilinear_sym
from libc.math cimport exp, log
cimport numpy as _np


cdef _np.ndarray[double, ndim=2] calculate_rho_rb(_np.ndarray[double, ndim=2] samples,
                                                  density, live_components):
    '''Calculate the responsibilities using the full Rao-Blackwellized scheme.'''
    rho = _np.zeros(( len(samples),len(density.components) ))
    density.multi_evaluate(samples, individual=rho, components=live_components)

    cdef:
        size_t n
        int    k
        double tiny = _np.finfo('d').tiny
        double [:] component_weights = density.weights
        double [:] log_denominator = logsumexp2D(rho, density.weights)
        double [:,:] memview_rho = rho

    for k in live_components:
        for n in range(len(samples)):
            memview_rho[n, k]  = exp(memview_rho[n, k]) * component_weights[k]
            # + "tiny" --> avoid division by zero
            memview_rho[n, k] /= exp(log_denominator[n]) + tiny

    return rho

cdef _np.ndarray[double, ndim=2] calculate_rho_non_rb(_np.ndarray[double, ndim=2] samples,
                                                      latent, density, live_components):
    '''Calculate the responsibilities using latent variables.'''
    rho = _np.zeros(( len(samples),len(density.components) ))
    for k in live_components:
        rho[latent==k,k] = 1.
    return rho

def _prepare_pmc_update(_np.ndarray[double, ndim=2] samples, weights, latent, mincount, density, rb, copy):
    """Check arguments of :py:func:`.gaussian_pmc` or
    :py:func:`.student_t_pmc` for contradictions, call the correct
    function to calculate ``rho`` depending on ``rb`` (Rao-Blackwell)
    and (if ``latent`` is not None) prune components which have less
    than ``mincount`` samples. Make a copy of the ``density`` if
    ``copy`` set to True. Else it is updated in place.

    Return ``density``, ``rho``, ``weight_normalization``,
    ``live_components`` and ``need_renormalize``.

    """
    need_renormalize = False

    if copy:
        density = _cp(density)

    if weights is not None:
        weights = _np.asarray(weights)
        assert len(weights.shape) == 1, 'Weights must be one-dimensional.'
        assert len(weights) == len(samples), \
            "Number of weights (%s) does not match the number of samples (%s)." % (len(weights), len(samples))
        weight_normalization = weights.sum()
    else:
        weight_normalization = float(len(samples))

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

        rho = calculate_rho_rb(samples, density, live_components)


    else: # if latent is not None
        count = _np.histogram(latent, bins=len(density.components), range=(0,len(density.components)))[0]

        # set up list of live_components
        live_components = []
        for k in range(len(density)):
            if density.weights[k] == 0.:
                continue
            live_components.append(k)

        if rb:
            rho = calculate_rho_rb(samples, density, live_components)
        else:
            rho = calculate_rho_non_rb(samples, latent, density, live_components)

        # prune components with less samples than ``mincount`` AFTER rho has been calculated
        for k in live_components:
            if count[k] < mincount:
                live_components.remove(k)
                density.weights[k] = 0.
                # when a component is pruned, the other weights must be renormalized
                need_renormalize = True
                print("Component %i died because of too few (%i) samples." %(k, count[k]))

    return density, rho, weight_normalization, live_components, need_renormalize


def gaussian_pmc(_np.ndarray[double, ndim=2] samples not None, density,
                 weights=None, latent=None, rb=True, mincount=0, copy=True):
    '''
    Adapt a Gaussian mixture ``density`` using the (M-)PMC algorithm
    according to [Cap+08]_. Another useful reference is [Kil+09]_.

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
    density, rho, weight_normalization, live_components, need_renormalize = \
        _prepare_pmc_update(samples, weights, latent, mincount, density, rb, copy)

    # -------------- update equations according to (14) in [Cap+08] --------------

    # allocate memory for covariances (other memory is allocated on demand)
    cov = _np.empty(( len(density.components),len(samples[0]),len(samples[0]) ))
    x_minus_mu = _np.empty((len(samples), len(samples[0])))


    if weights is not None:

        # new component weights
        alpha  = _np.einsum('n,nk->k', weights, rho)
        inv_unnormalized_alpha = 1. / regularize(alpha)
        alpha /= weight_normalization

        # new means
        mu = _np.einsum('n,nk,ni->ki', weights, rho, samples)
        mu = _np.einsum('ki,k->ki', mu, inv_unnormalized_alpha)

        # new covars
        for k in live_components:
            x_minus_mu[:] = samples
            x_minus_mu -= mu[k]
            _np.einsum('n,n,ni,nj->ij', weights, rho[:,k], x_minus_mu, x_minus_mu, out=cov[k])
            cov[k] *= inv_unnormalized_alpha[k]

    else: # if weights is None

        # new component weights
        alpha  = _np.einsum('nk->k', rho)
        inv_unnormalized_alpha = 1. / regularize(alpha)
        alpha /= weight_normalization

        # new means
        mu = _np.einsum('nk,ni->ki', rho, samples)
        mu = _np.einsum('ki,k->ki', mu, inv_unnormalized_alpha)

        # new covars
        for k in live_components:
            x_minus_mu[:] = samples
            x_minus_mu -= mu[k]
            _np.einsum('n,ni,nj->ij', rho[:,k], x_minus_mu, x_minus_mu, out=cov[k])
            cov[k] *= inv_unnormalized_alpha[k]

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

cdef class _DOFCondition(object):
    '''Implements the first order condition for the degree of freedom of
    a StudentT mixture: The member function :py:meth:`.ccall` must
    evalutate to zero. This means, a root finder should be run on
    :py:meth:`.ccall`.

    .. seealso::
        equation (16) in [HOD12]_

    :param const:

        Double; the equation's part which does not depend on ``nu`` and
        thus only needs to be calculated once in advance

    '''
    cdef double const
    def __cinit__(self, double const):
        self.const = const
    def __call__ (self, double nu):
        return self.const + log(.5 * nu) - _psi(.5 * nu)

def student_t_pmc(_np.ndarray[double, ndim=2] samples not None, density, weights=None,
                  latent=None, rb=True, dof_solver_steps=100, mindof=1e-5, maxdof=1e3, mincount=0, copy=True):
    '''
    Adapt a Student t mixture ``density`` using the (M-)PMC algorithm
    according to [Cap+08]_ and [HOD12]_.Another useful reference is
    [Kil+09]_.

    :param samples:

        Matrix-like array; the samples to be used for the PMC run.

    :param density:

        :py:class:`.MixtureDensity` with :py:class:`.StundentT` components;
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

    :param dof_solver_steps:

        Integer; If ``0``, the Student t's degrees of freedom are not updated,
        otherwise an iterative algorithm is run for at most ``dof_solver_steps`` steps.

        .. note::

            There is no closed form solution for the optimal degree of
            freedom. If ``dof_solver_steps`` is not ``0``, ``len(density)`` first order
            equations must be solved numerically which can take a while.

    :param mindof, maxdof:

        Float; Degree of freedom adaptation is a one dimentional root
        finding problem. The numerical root finder used in this function
        (:py:func:`scipy.optimize.brentq`) needs an interval where to
        search.

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
    density, rho, weight_normalization, live_components, need_renormalize = \
        _prepare_pmc_update(samples, weights, latent, mincount, density, rb, copy)

    # -------------- update equations according to (14) in [Cap+08] --------------

    cdef size_t dim = len(samples[0]), N = len(samples), K = len(density)

    # allocate memory for covariances (other memory is allocated on demand)
    cov = _np.empty( (K,dim,dim) )
    x_minus_mu = _np.empty( (N,dim) )

    # predefine variables for calculation of gamma
    cdef:
        size_t n, i
        int    k
        double double_dim = dim
        double old_nu_k
        double [:] x_minus_mu_n_k = x_minus_mu[0]
        _np.ndarray[double, ndim=1] old_mu_k = _np.empty(dim)
        double [:] view_old_mu_k = old_mu_k
        _np.ndarray[double, ndim=2] old_inv_sigma_k = _np.empty( (dim,dim) )
        double [:,:] view_old_inv_sigma_k = old_inv_sigma_k
        double [:,:] gamma = _np.empty( (N,K) )
        double [:,:] samples_memview = samples
        double [:,:] rho_memview = rho

    # calculate gamma
    for k in live_components:
        old_nu_k           = density.components[k].dof
        old_mu_k[:]        = density.components[k].mu
        old_inv_sigma_k[:] = density.components[k].inv_sigma
        for n in range(N):
            for i in range(dim):
                x_minus_mu_n_k[i]  = samples_memview[n,i]
                x_minus_mu_n_k[i] -= view_old_mu_k[i]
            gamma[n,k] = (old_nu_k + double_dim) / (old_nu_k + bilinear_sym(view_old_inv_sigma_k, x_minus_mu_n_k) )

    if weights is not None:

        # new component weights
        alpha  = _np.einsum('n,nk->k', weights, rho)
        inv_unnormalized_alpha = 1. / regularize(alpha)
        alpha /= weight_normalization

        # new means
        mu = _np.einsum('n,nk,nk,ni->ki', weights, rho, gamma, samples)
        mu_normalization = _np.einsum('n,nk,nk->k', weights, rho, gamma)
        mu_normalization = 1. / regularize(mu_normalization)
        mu = _np.einsum('ki,k->ki', mu, mu_normalization)

        # new covars
        for k in live_components:
            x_minus_mu[:] = samples
            x_minus_mu -= mu[k]
            _np.einsum('n,n,n,ni,nj->ij', weights, rho[:,k], gamma[:,k], x_minus_mu, x_minus_mu, out=cov[k])
            cov[k] *= inv_unnormalized_alpha[k]

    else: # if weights is None

        # new component weights
        alpha  = _np.einsum('nk->k', rho)
        inv_unnormalized_alpha = 1. / regularize(alpha)
        alpha /= weight_normalization

        # new means
        mu = _np.einsum('nk,nk,ni->ki', rho, gamma, samples)
        mu_normalization = _np.einsum('nk,nk->k', rho, gamma)
        mu_normalization = 1. / regularize(mu_normalization)
        mu = _np.einsum('ki,k->ki', mu, mu_normalization)

        # new covars
        for k in live_components:
            x_minus_mu[:] = samples
            x_minus_mu -= mu[k]
            _np.einsum('n,n,ni,nj->ij', rho[:,k], gamma[:,k], x_minus_mu, x_minus_mu, out=cov[k])
            cov[k] *= inv_unnormalized_alpha[k]

    cdef double nu_condition_const, bilinear

    if dof_solver_steps:

        new_dof = [-1 for k in range(K)]

        # calculate xi_n_k + delta_n_k from [HOD12] --> can use ``gamma`` buffer because it is not needed any more
        for k in live_components:

            old_nu_k           = density.components[k].dof
            old_mu_k[:]        = density.components[k].mu
            old_inv_sigma_k[:] = density.components[k].inv_sigma

            for n in range(N):
                for i in range(dim):
                    x_minus_mu_n_k[i]  = samples_memview[n,i]
                    x_minus_mu_n_k[i] -= view_old_mu_k[i]
                bilinear = bilinear_sym(view_old_inv_sigma_k, x_minus_mu_n_k)

                # xi
                gamma[n,k]  =  log(.5 * (bilinear   + old_nu_k) )
                gamma[n,k] -= _psi(.5 * (double_dim + old_nu_k) )
                gamma[n,k] *= rho_memview[n,k]
                gamma[n,k] += (1. - rho_memview[n,k]) * ( log(.5 * old_nu_k) - _psi(.5 * old_nu_k) )

                # delta
                gamma[n,k] += rho_memview[n,k] * (double_dim + old_nu_k) / (bilinear + old_nu_k)
                gamma[n,k] += (1. - rho_memview[n,k])

            # gamma should now be xi + delta

        for k in live_components:

            # calculate constant part of condition for nu
            if weights is None:
                nu_condition_const  = _np.einsum('n->', gamma[:,k])
            else:
                nu_condition_const  = _np.einsum('n,n->', gamma[:,k], weights)
            nu_condition_const /= weight_normalization
            nu_condition_const  = 1. - nu_condition_const
            nu_condition = _DOFCondition(nu_condition_const)

            # solve the first-order condition
            try:
                new_dof[k] = _find_root(nu_condition, mindof, maxdof, maxiter=dof_solver_steps)
            except RuntimeError: # occurs if not converged
                print("WARNING: ``dof`` solver for component %i did not converge." % k)
                new_dof[k] = density.components[k].dof
            except ValueError as error:
                # check if nu_condition has the same sign at mindof and maxdof
                # Note: nu_condition is a decreasing function
                #   - if nu_condition(mindof) < 0. we know the root is < mindof --> set it mindof
                #   - if nu_condition(maxdof) > 0. we know the root is > maxdof --> set it maxdof
                if nu_condition(mindof) < 0.:
                    new_dof[k] = mindof
                elif nu_condition(maxdof) > 0.:
                    new_dof[k] = maxdof
                else:
                    raise RuntimeError('``dof`` adaptation for component %i raised an error.'% k, error)

    else: # if not dof

        new_dof = [c.dof for c in density.components]

    # ----------------------------------------------------------------------------

    # apply the updated mixture weights, means and covariances
    for k in live_components:
        component = density.components[k]
        density.weights[k] = alpha[k]
        # if matrix is not positive definite, the update will fail
        # in that case replug the old values and set its weight to zero
        old_mu    = component.mu    # do not need to copy because .update creates a new array
        old_sigma = component.sigma # do not need to copy because .update creates a new array
        old_dof   = component.dof
        try:
            component.update(mu[k], cov[k], new_dof[k])
        except _np.linalg.LinAlgError:
            print("Could not update component %i --> weight is set to zero." % k)
            component.update(old_mu, old_sigma, old_dof)
            density.weights[k] = 0.
            # when a component is pruned, the other weights must be renormalized
            need_renormalize = True

    if need_renormalize:
        density.normalize()

    return density
