"""Variational clustering as described in [Bis06]_

"""

from __future__ import division, print_function

import numpy as _np
from scipy.special import gamma as _gamma
from scipy.special import gammaln as _gammaln
from scipy.special.basic import digamma as _digamma

from ..density.gauss import Gauss
from ..density.mixture import MixtureDensity, recover_gaussian_mixture as _unroll
from ..tools._doc import _inherit_docstring, _add_to_docstring
from ..tools._regularize import regularize

cimport numpy as _np
from libc.math cimport exp, log
from pypmc.tools._linalg cimport bilinear_sym, chol_inv_det

import logging
logger = logging.getLogger(__name__)

DTYPE = _np.float64
ctypedef double DTYPE_t

class GaussianInference(object):
    '''Approximate a probability density by a Gaussian mixture with a variational
    Bayes approach. The motivation, notation, and derivation is explained in
    detail in chapter 10.2 in [Bis06]_.

    Typical usage: call :meth:`run` until convergence. If interested
    in clustering/classification, extract the responsibility matrix as
    the attribute ``r``. Else get the Gaussian mixture density at the
    mode of the variational posterior using :meth:`make_mixture`.

    .. seealso ::

        Another implementation can be found at https://github.com/jamesmcinerney/vbmm.

    :param data:

        Matrix like array; Each of the :math:`N` rows contains one
        :math:`D`-dimensional sample from the probability density to be
        approximated.

    :param components:

        Integer; :math:`K` is the number of Gaussian components in the
        approximating Gaussian mixture. Will be detected from
        ``initial_guess`` if provided.

    :param weights:

        Vector-like array; The i-th of the :math:`N` entries contains the
        weight of the i-th sample in ``data``. Weights must be nonnegative and finite.

    :param initial_guess:

        string or :py:class:`pypmc.density.mixture.MixtureDensity` with Gaussian
        (:py:class:`pypmc.density.gauss.Gauss`) components;

        Allowed string values:

            * "first": initially place the components (defined by the mean
              parameter ``m``) at the first ``K`` data points.
            * "random": like "first", but randomly select ``K`` data points. For
              reproducibility, set the seed with ``numpy.random.seed(123)``

        If a `MixtureDensity`, override other (default) values of the parameters
        ``m``, ``W`` and ``alpha``.

        Default: "first"

    All keyword arguments are processed by :py:meth:`set_variational_parameters`.

    '''

    def __init__(self, _np.ndarray data, int components=0, weights=None, initial_guess="first", **kwargs):
        self.N = data.shape[0]
        if data.ndim == 1:
            self.data = data.reshape(self.N, 1)
        else:
            self.data = data
        self.dim = self.data.shape[1]
        if weights is not None:
            assert weights.shape == (self.N,), \
                    "The number of samples (%s) does not match the number of weights (%s)" %(self.N, weights.shape[0])
            assert _np.isfinite(weights).all(), \
                    'Some weights are not finite; i.e., inf or nan\n' + str(weights)
            # normalize weights to N (not one)
            sum_w = weights.sum()
            assert sum_w > 0, 'Sum of weights <= 0 (%g)' % sum_w
            self.weights = self.N * (weights / sum_w)

            # use weighted update formulae
            self._update_N_comp = self._update_N_comp_weighted
            self._update_x_mean_comp = self._update_x_mean_comp_weighted
            self._update_S = self._update_S_weighted
            self._update_expectation_log_q_Z = self._update_expectation_log_q_Z_weighted

        self._initialize_K(initial_guess, components, kwargs)

        # if `initial_guess` is a string, it is used to set for example m
        self.set_variational_parameters(initial_guess=initial_guess, **kwargs)

        if not isinstance(initial_guess, str):
            self._parse_initial_guess(initial_guess)

        self._initialize_intermediate(self.N)

        # compute expectation values for the initial parameter values
        # so a valid bound can be computed after object is initialized
        self.E_step()

    def E_step(self):
        '''Compute expectation values and summary statistics.'''

        # check ln_lambda first to catch an invalid W matrix
        # before expensive loop over samples
        self._update_expectation_det_ln_lambda()
        self._update_expectation_gauss_exponent()
        self._update_expectation_ln_pi()
        self._update_r()
        self._update_N_comp()
        self._update_x_mean_comp()
        self._update_S()

    def M_step(self):
        '''Update parameters of the Gaussian-Wishart distribution.'''

        self.nu = self.nu0 + self.N_comp
        self.alpha = self.alpha0 + self.N_comp
        self.beta = self.beta0 + self.N_comp
        self._update_m()
        self._update_W()

    def make_mixture(self):
        '''Return the mixture-density defined by the
        mode of the variational-Bayes estimate.

        '''

        # find mode of Gaussian-Wishart distribution
        # and invert to find covariance. The result
        # \Lambda_k = (\nu_k - D) W_k
        # the mode of the Gauss-Wishart exists only if \nu_k > D
        # turns out to be independent of beta.

        # The most likely value of the mean is m_k,
        # the mean parameter of the Gaussian q(\mu_k).

        # The mode of the Dirichlet exists only if \alpha_k > 1

        components = []
        weights = []
        skipped = []
        for k, W in enumerate(self.W):
            # Dirichlet mode
            # do not divide by the normalization constant because:
            #   1. this will be done automatically by the mixture contructor
            #   2. in case \alpha_k < 1 and normalization = \sum_{n=1}^{N}\(alpha_k-1) < 0
            #      the results would be nonsense but no warning would be printed
            #      because in that case \frac{\alpha_k - 1}{normalization} > 0
            pi = self.alpha[k] - 1.
            if pi <= 0:
                logger.warning("Skipped component %i because of zero weight" %k)
                skipped.append(k)
                continue

            # Gauss-Wishart mode
            if self.nu[k] <= self.dim:
                logger.warning("Gauss-Wishart mode of component %i is not defined" %k)
                skipped.append(k)
                continue

            try:
                W = (self.nu[k] - self.dim) * W
                cov = chol_inv_det(W)[1]
                components.append(Gauss(self.m[k], cov))
            except Exception as error:
                logger.error("Could not create component %i. The error was: %s" %(k, repr(error)))
                skipped.append(k)
                continue

            # relative weight properly normalized
            weights.append(pi)

        if skipped:
            logger.warning("The following components have been skipped:", skipped)

        return MixtureDensity(components, weights)

    def likelihood_bound(self):
        '''Compute the lower bound on the true log marginal likelihood
        :math:`L(Q)` given the current parameter estimates.

        '''

        # todo easy to parallelize sum of independent terms
        bound  = self._update_expectation_log_p_X()
        bound += self._update_expectation_log_p_Z()
        bound += self._update_expectation_log_p_pi()
        bound += self._update_expectation_log_p_mu_lambda()
        bound -= self._update_expectation_log_q_Z()
        bound -= self._update_expectation_log_q_pi()
        bound -= self._update_expectation_log_q_mu_lambda()

        return bound

    def posterior2prior(self):
        '''Return references to posterior values of all variational parameters
        as dict.

        .. hint::
            :py:class:`.GaussianInference`\ (`..., **output`) creates a new
            instance using the inferred posterior as prior.

        '''
        return dict(alpha0=self.alpha.copy(), beta0=self.beta.copy(), nu0=self.nu.copy(),
                    m0=self.m.copy(), W0=self.W.copy(), components=self.K)

    def prior_posterior(self):
        '''Return references to prior and posterior values of all variational
        parameters as dict.

        '''

        return dict(alpha0=self.alpha0.copy(), beta0=self.beta0.copy(), m0=self.m0.copy(),
                    nu0=self.nu0.copy(), W0=self.W0.copy(), alpha=self.alpha.copy(), beta=self.beta.copy(),
                    m=self.m.copy(), nu=self.nu.copy(), W=self.W.copy(), components=self.K)

    def prune(self, threshold=1.):
        r'''Delete components with an effective number of samples
        :math:`N_k` below the threshold.

        :param threshold:

            Float; the minimum effective number of samples a component must have
            to survive.

        '''

        # nothing to do for a zero threshold
        if not threshold:
            return

        components_to_survive = _np.where(self.N_comp >= threshold)[0]
        K = len(components_to_survive)
        if K == 0:
            raise ValueError("Prune threshold %g too large, would remove all components" % threshold)
        self.K = K

        # list all vector and matrix vmembers
        vmembers = ('alpha0', 'alpha', 'beta0', 'beta', 'expectation_det_ln_lambda',
                    'expectation_ln_pi', 'N_comp', 'nu0', 'nu', 'm0', 'm', 'S', 'W0', 'inv_W0', 'W',
                    'log_det_W', 'log_det_W0', 'x_mean_comp')
        mmembers = ('expectation_gauss_exponent', 'log_rho', 'r')

        # shift surviving across dead components
        k_new = 0
        for k_old in components_to_survive:
            # reindex surviving components
            if k_old != k_new:
                for m in vmembers:
                    m = getattr(self, m)
                    m[k_new] = m[k_old]
                for m in mmembers:
                    m = getattr(self, m)
                    m[:, k_new] = m[:, k_old]

            k_new += 1

        # cut the unneccessary part of the data
        for m in vmembers:
            setattr(self, m, getattr(self, m)[:self.K])
        for m in mmembers:
            setattr(self, m, getattr(self, m)[:, :self.K])

        # recreate consistent expectation values
        self.E_step()

    def run(self, iterations=1000, prune=1., rel_tol=1e-10, abs_tol=1e-5, verbose=False):
        r'''Run variational-Bayes parameter updates and check for convergence
        using the change of the log likelihood bound of the current and the last
        step. Convergence is not declared if the number of components changed,
        or if the bound decreased. For the standard algorithm, the bound must
        increase, but for modifications, this useful property may not hold for
        all parameter values.

        Return the number of iterations at convergence, or None.

        :param iterations:
            Maximum number of updates.

        :param prune:
            Call :py:meth:`prune` after each update; i.e., remove components
            whose associated effective number of samples is below the
            threshold. Set ``prune=0`` to deactivate.
            Default: 1 (effective samples).

        :param rel_tol:
            Relative tolerance :math:`\epsilon`. If two consecutive values of
            the log likelihood bound, :math:`L_t, L_{t-1}`, are close, declare
            convergence. More precisely, check that

            .. math::
                \left\| \frac{L_t - L_{t-1}}{L_t} \right\| < \epsilon .

        :param abs_tol:
            Absolute tolerance :math:`\epsilon_{a}`. If the current bound
            :math:`L_t` is close to zero, (:math:`L_t < \epsilon_{a}`), declare
            convergence if

            .. math::
                \| L_t - L_{t-1} \| < \epsilon_a .

        '''
        if verbose:
            from pypmc.tools.util import depr_warn_verbose
            depr_warn_verbose( __name__)

        old_K = None
        for i in range(1, iterations + 1):
            # recompute bound in 1st step or if components were removed
            if self.K == old_K:
                old_bound = bound
            else:
                old_bound = self.likelihood_bound()
                logger.info('New bound=%g, K=%d, N_k=%s' % (old_bound, self.K, self.N_comp))

            self.update()
            bound = self.likelihood_bound()

            logger.info('After update %d: bound=%.15g, K=%d, N_k=%s' % (i, bound, self.K, self.N_comp))

            if bound < old_bound:
                logger.warning('Bound decreased from %g to %g' % (old_bound, bound))

             # exact convergence
            if bound == old_bound:
                return i
            # approximate convergence
            # but only if bound increased
            diff = bound - old_bound
            if diff > 0:
                # handle case when bound is close to 0
                if abs(bound) < abs_tol:
                    if abs(diff) < abs_tol:
                        return i
                else:
                    if abs(diff / bound) < rel_tol:
                        return i

            # save K *before* pruning
            old_K = self.K
            self.prune(prune)
        # not converged
        return None

    def set_variational_parameters(self, *args, **kwargs):
        r'''Reset the parameters to the submitted values or default.

        Use this function to set the prior value (indicated by the
        subscript :math:`0` as in :math:`\alpha_0`) or the initial
        value (e.g., :math:`\alpha`) used in the iterative procedure
        to find the values of the hyperparameters of variational
        posterior distribution.

        Every parameter can be set in two ways:

        1. It is specified for only one component, then it is copied
        to all other components.

        2. It is specified separately for each component as a
        :math:`K` vector.

        The prior and posterior variational distributions of
        :math:`\boldsymbol{\mu}` and :math:`\boldsymbol{\Lambda}` for
        each component are given by

        .. math::

            q(\boldsymbol{\mu}, \boldsymbol{\Lambda}) =
            q(\boldsymbol{\mu}|\boldsymbol{\Lambda}) q(\boldsymbol{\Lambda}) =
            \prod_{k=1}^K
              \mathcal{N}(\boldsymbol{\mu}_k|\boldsymbol{m_k},(\beta_k\boldsymbol{\Lambda}_k)^{-1})
              \mathcal{W}(\boldsymbol{\Lambda}_k|\boldsymbol{W_k}, \nu_k),

        where :math:`\mathcal{N}` denotes a Gaussian and
        :math:`\mathcal{W}` a Wishart distribution. The weights
        :math:`\boldsymbol{\pi}` follow a Dirichlet distribution

        .. math::
                q(\boldsymbol{\pi}) = Dir(\boldsymbol{\pi}|\boldsymbol{\alpha}).

        .. warning ::

            This function may delete results obtained by :py:meth:`.update`.

        :param alpha0, alpha:

            Float or :math:`K` vector; parameter of the mixing
            coefficients' probability distribution (prior:
            :math:`\alpha_0`, posterior initial value: :math:`\alpha`).

            .. math::
                \alpha_i > 0, i=1 \dots K.

            A scalar is promoted to a :math:`K` vector as

            .. math::
                \boldsymbol{\alpha} = (\alpha,\dots,\alpha),

            but a `K` vector is accepted, too.

            Default:

            .. math::
                \alpha = 10^{-5}.

        :param beta0, beta:

            Float or :math:`K` vector; :math:`\beta` parameter of
            the probability distribution of :math:`\boldsymbol{\mu}`
            and :math:`\boldsymbol{\Lambda}`. The same restrictions
            as for ``alpha`` apply. Default:

            .. math::
                \beta_0 = 10^{-5}.

        :param nu0, nu:

            Float or :math:`K` vector; degrees of freedom of the
            Wishart distribution of :math:`\boldsymbol{\Lambda}`.
            A well defined Wishard distribution requires:

            .. math::
                \nu_0 \geq D - 1.

            The same restrictions as for ``alpha`` apply.

            Default:

            .. math::
                \nu_0 = D - 1 + 10^{-5}.

        :param m0, m:

            :math:`D` vector or :math:`K \times D` matrix; mean
            parameter for the Gaussian
            :math:`q(\boldsymbol{\mu_k}|\boldsymbol{m_k}, \beta_k
            \Lambda_k)`.

            Default:

            For the prior of each component:

            .. math::
                \boldsymbol{m}_0 = (0,\dots,0)

            For initial value of the posterior,
            :math:`\boldsymbol{m}`: the sequence of :math:`K \times D`
            equally spaced values in [-1,1] reshaped to :math:`K
            \times D` dimensions.

            .. warning:: If all :math:`\boldsymbol{m}_k` are identical
                initially, they may remain identical. It is advisable
                to randomly scatter them in order to avoid singular
                behavior.

        :param W0, W:

            :math:`D \times D` or :math:`K \times D \times D`
            matrix-like array; :math:`\boldsymbol{W}` is a symmetric
            positive-definite matrix used in the Wishart distribution.
            Default: identity matrix in :math:`D` dimensions for every
            component.

        '''
        if args: raise TypeError('keyword args only')

        self.alpha0 = kwargs.pop('alpha0', 1e-5)
        if not _np.iterable(self.alpha0):
            self.alpha0 =  self.alpha0 * _np.ones(self.K)
        else:
            self.alpha0 = _np.array(self.alpha0)
        self._check_K_vector('alpha0')
        self.alpha = kwargs.pop('alpha', _np.ones(self.K) * self.alpha0)
        self._check_K_vector('alpha')
        self.alpha = _np.array(self.alpha)

        # in the limit beta --> 0: uniform prior
        self.beta0 = kwargs.pop('beta0', 1e-5)
        if not _np.iterable(self.beta0):
            self.beta0 =  self.beta0 * _np.ones(self.K)
        else:
            self.beta0 = _np.array(self.beta0)
        self._check_K_vector('beta0')
        self.beta = kwargs.pop('beta', _np.ones(self.K) * self.beta0)
        self._check_K_vector('beta')
        self.beta = _np.array(self.beta)

        # smallest possible nu such that the Wishart pdf does not diverge at 0 is self.dim + 1
        # smallest possible nu such that the Gauss-Wishart pdf does not diverge is self.dim
        # allowed values: nu > self.dim - 1
        nu_min = self.dim - 1.
        self.nu0 = kwargs.pop('nu0', nu_min + 1e-5)
        if not _np.iterable(self.nu0):
            self.nu0 = self.nu0 * _np.ones(self.K)
        else:
            self.nu0 = _np.array(self.nu0)
        self._check_K_vector('nu0', min=nu_min)
        self.nu = kwargs.pop('nu', self.nu0 * _np.ones(self.K))
        self._check_K_vector('nu', min=nu_min)
        self.nu = _np.array(self.nu)

        self.m0 = _np.array( kwargs.pop('m0', _np.zeros(self.dim)) )
        if len(self.m0) == self.dim:
            # vector or matrix?
            if len(self.m0.shape) == 1:
                self.m0 = _np.vstack(tuple([self.m0] * self.K))

        initial_guess = kwargs.pop('initial_guess')

        # If the initial means are identical, the K remain identical in all updates.
        self.m = kwargs.pop('m', None)
        if self.m is None:
            if isinstance(initial_guess, str):
                self.m = self._initialize_m(initial_guess)
            else:
                # old default, can be really bad
                # should be overwritten later!
                self.m = _np.linspace(-1.,1., self.K*self.dim).reshape((self.K, self.dim))
        else:
            self.m = _np.array(self.m)
        for name in ('m0', 'm'):
            if getattr(self, name).shape != (self.K, self.dim):
                raise ValueError('Shape of %s %s does not match (K,d)=%s' % (name, getattr(self, name).shape, (self.K, self.dim)))

        # covariance matrix; unit matrix <--> unknown correlation
        self.W0     = kwargs.pop('W0', None)
        if self.W0 is None:
            self.W0     = _np.eye(self.dim)
            self.inv_W0 = self.W0.copy()
            log_det = 0.0
        elif self.W0.shape == (self.dim, self.dim):
            self.W0     = _np.array(self.W0)
            self.inv_W0, log_det = chol_inv_det(self.W0)[1:]
        # handle both above cases
        if self.W0.shape == (self.dim, self.dim):
            self.W0 = _np.array([self.W0] * self.K)
            self.inv_W0 = _np.array([self.inv_W0] * self.K)
            self.log_det_W0 = _np.array([log_det] * self.K)
        # full sequence of matrices given
        elif self.W0.shape == (self.K, self.dim, self.dim):
            self.log_det_W0 = _np.empty((self.K,))
            self.inv_W0 = _np.empty_like(self.W0)
            for k in range(self.K):
                self.inv_W0[k], self.log_det_W0[k] = chol_inv_det(self.W0[k])[1:]
        else:
            raise ValueError('W0 is neither None, nor a %s array, nor a %s array.' % ((self.dim, self.dim), (self.K, self.dim, self.dim)))
        self.W = kwargs.pop('W', self.W0.copy())
        if self.W.shape != (self.K, self.dim, self.dim):
            raise ValueError('Shape of W %s does not match (K, d, d)=%s' % (self.W.shape, (self.K, self.dim, self.dim)))
        # check if W valid covariance and compute determinant
        self.log_det_W = _np.array([chol_inv_det(W)[2] for W in self.W])

        if kwargs: raise TypeError('unexpected keyword(s): ' + str(kwargs.keys()))

    def update(self):
        '''Recalculate the parameters (M step) and expectation values (E step)
        using the update equations.

        '''

        self.M_step()
        self.E_step()

    def _check_initial_guess(self, initial_guess, other_args):
        if 'm' in other_args:
            raise ValueError('Specify EITHER ``m`` OR ``initial_guess``')
        if 'W' in other_args:
            raise ValueError('Specify EITHER ``W`` OR ``initial_guess``')
        if 'alpha' in other_args:
            raise ValueError('Specify EITHER ``alpha`` OR ``initial_guess``')
        if 'beta' in other_args:
            raise ValueError('Specify EITHER ``beta`` OR ``initial_guess``')
        if 'nu' in other_args:
            raise ValueError('Specify EITHER ``nu`` OR ``initial_guess``')

    def _initialize_K(self, initial_guess, components, kwargs):
        if not isinstance(initial_guess, str):
            self.K = len(initial_guess)
            self._check_initial_guess(initial_guess, kwargs)
        elif components > 0:
            self.K = components
        else:
            raise ValueError('Specify either `components` or a mixture density as `initial_guess` to set the initial values')

    def _check_K_vector(self, name, min=0.0):
        v = getattr(self, name)
        if len(v.shape) != 1:
            raise ValueError('%s is not a vector but has shape %s' % (name, v.shape))
        if len(v) != self.K:
            raise ValueError('len(%s)=%d does not match K=%d' % (name, len(v), self.K))
        if not (v > min).all():
            raise ValueError('All elements of %s must exceed %g. %s=%s' % (name, min, name, v))

    def _initialize_m(self, initial_guess):
        '''Provide initial guess for ``m`` depending on chosen method in the string ``initial_guess``.'''

        if self.K > self.N:
            raise ValueError("Can't auto-initialize ``m`` with more output components than samples."
                             " Specify ``m`` explicitly.")

        if initial_guess == 'first':
            # the idea behind this is to close to at least one datum such that
            # components do not die in the very first step only because of a bad
            # initialization
            return self.data[:self.K].copy()
        elif initial_guess == 'random':
            return self.data[_np.random.choice(self.N, size=self.K, replace=False)].copy()
        else:
            raise ValueError('Invalid ``initial_guess``: ' + str(initial_guess))

    def _initialize_intermediate(self, N_samples):
        '''Create all intermediate quantities needed for the iteration in ``self.update``.
        :param N_samples:

            Int; The effective number of samples for which, for example, the responsibility
            needs to be computed. Note that this may differ from ``self.N`` for `:class:VBMerge:`.

        '''

        self.expectation_gauss_exponent = _np.zeros((N_samples, self.K))
        self.log_rho = _np.zeros_like(self.expectation_gauss_exponent)
        self.r = _np.zeros_like(self.expectation_gauss_exponent)

        self.x_mean_comp = _np.zeros((self.K, self.dim))
        self.S = _np.empty_like(self.W)
        self.N_comp = _np.zeros(self.K)
        self.expectation_det_ln_lambda = _np.zeros_like(self.N_comp)
        self.expectation_ln_pi = _np.zeros_like(self.N_comp)

    def _parse_initial_guess(self, initial_guess):
        means, covs, component_weights = _unroll(initial_guess)
        N = self.N
        K = self.K

        # solve Dirichlet mode as fct. of alpha
        c_alpha = self.alpha0.sum() + N
        self.alpha = component_weights * (c_alpha - K) + 1

        # beta_0 + N_k
        self.beta = self.beta0 + N * component_weights

        # nu_0 + N_k
        self.nu = self.nu0 + N * component_weights

        assert (self.alpha > 0.0).all()
        assert (self.beta > 0.0).all()
        assert (self.nu > self.dim - 1).all()

        self.m = means
        self.W = _np.empty_like(covs)
        for k in range(self.K):
            # modify covariance in-place OK, as _unroll returns copies
            covs[k] *= (self.nu[k] - self.dim)
            self.W[k], self.log_det_W[k] = chol_inv_det(covs[k])[1:]

        # det(W) = det(Cov^-1)
        self.log_det_W *= -1

    def _update_log_rho(self):
        # (10.46)

        cdef:
            DTYPE_t dlog = self.dim * log(2. * _np.pi)
            DTYPE_t [:] expectation_ln_pi = self.expectation_ln_pi
            DTYPE_t [:] expectation_det_ln_lambda = self.expectation_det_ln_lambda
            DTYPE_t [:,:] log_rho = self.log_rho
            DTYPE_t [:,:] expectation_gauss_exponent = self.expectation_gauss_exponent

            size_t K = self.K
            size_t N = len(log_rho)
            size_t k,n

        for n in range(N):
            for k in range(K):
                log_rho[n,k] = expectation_ln_pi[k] + 0.5 * (expectation_det_ln_lambda[k] - dlog - expectation_gauss_exponent[n,k])

    def _update_m(self):
        # (10.61)

        for k in range(self.K):
            self.m[k] = 1. / self.beta[k] * (self.beta0[k] * self.m0[k] + self.N_comp[k] * self.x_mean_comp[k])

    def _update_N_comp_weighted(self):
        # modified (10.51)

        _np.einsum('n,nk->k', self.weights, self.r, out=self.N_comp)
        self.inv_N_comp = 1. / regularize(self.N_comp)

    def _update_N_comp(self):
        # (10.51)

        _np.einsum('nk->k', self.r, out=self.N_comp)
        self.inv_N_comp = 1. / regularize(self.N_comp)

    def _update_r(self):
        # (10.49)

        self._update_log_rho()

        cdef:
            DTYPE_t [:,:] log_rho = self.log_rho
            DTYPE_t [:,:] r = self.r

            DTYPE_t max = 0.0
            DTYPE_t tiny = _np.finfo('d').tiny
            DTYPE_t norm, norm_inv, log_norm_inv

            cdef size_t K = self.K
            cdef size_t N = len(log_rho)
            cdef size_t k,n

        for n in range(N):
            max = log_rho[n,0]

            # find largest responsibility
            for k in range(1, K):
                if log_rho[n,k] > max:
                    max = log_rho[n,k]

            # rescale relative to largest value, so values in [0,1] (linear scale)
            # in the division, the extra scale factor drops out automagically
            # through the normalization
            norm = 0.0
            for k in range(K):
                log_rho[n,k] -= max
                r[n,k] = exp(log_rho[n,k])
                norm += r[n,k]

            # normalize to unity
            # MUL faster than DIV :)
            # store the properly normalized log(rho) for bound calculation
            norm_inv = 1. / norm
            log_norm_inv = log(norm_inv)
            for k in range(K):
                r[n,k] *= norm_inv
                # avoid overflows and nans when taking the log of 0
                if r[n,k] == 0.0:
                    r[n,k] = tiny
                log_rho[n,k] += log_norm_inv
        if not _np.isfinite(self.r).any():
            raise _np.linalg.LinAlgError('Encountered inf or nan in update of responsibilities\n' + str(self.r))

    def _update_expectation_det_ln_lambda(self):
        # (10.65)

        self.expectation_det_ln_lambda[:] = 0.0
        tmp = _np.zeros_like(self.nu)
        for i in range(1, self.dim + 1):
            tmp[:] = self.nu
            tmp += 1. - i
            tmp *= 0.5
            # digamma aware of vector input
            self.expectation_det_ln_lambda += _digamma(tmp)

        self.expectation_det_ln_lambda += self.dim * log(2.)
        self.expectation_det_ln_lambda += self.log_det_W

    def _update_expectation_gauss_exponent(self):
        # (10.64)

        cdef:
            DTYPE_t [:] beta = self.beta
            DTYPE_t [:,:] data = self.data
            DTYPE_t [:,:] m = self.m
            DTYPE_t [:] nu = self.nu
            DTYPE_t [:,:] expectation_gauss_exponent = self.expectation_gauss_exponent
            # use double directly to avoid error
            # Memoryview 'DTYPE_t[:]' not conformable to memoryview 'double[:]'.
            double  [:,:] W = _np.empty_like(self.W[0])
            double  [:] tmp  = _np.zeros(self.dim, dtype=DTYPE)
            size_t K = self.K
            size_t N = len(expectation_gauss_exponent)
            size_t dim = self.dim
            size_t k,n,i,j

        for k in range(K):
            W = self.W[k]
            for n in range(N):
                for i in range(dim):
                    tmp[i] = data[n,i] - m[k,i]

                expectation_gauss_exponent[n,k] = dim / beta[k] + nu[k] * bilinear_sym(W, tmp)

    def _update_expectation_ln_pi(self):
        # (10.66)

        self.expectation_ln_pi[:] = _digamma(self.alpha)
        self.expectation_ln_pi   -= _digamma(self.alpha.sum())

    def _update_x_mean_comp_weighted(self):
        # modified (10.52)

        cdef:
            DTYPE_t w
            DTYPE_t [:] inv_N_comp = self.inv_N_comp
            DTYPE_t [:] weights = self.weights
            DTYPE_t [:,:] r = self.r
            DTYPE_t [:,:] data = self.data
            DTYPE_t [:,:] x_mean_comp = self.x_mean_comp

            cdef size_t K = self.K
            cdef size_t N = len(r)
            cdef size_t dim = self.dim
            cdef size_t k,n,i

        x_mean_comp[:,:] = 0.0

        for k in range(K):
            for n in range(N):
                w = weights[n] * r[n,k]
                for i in range(dim):
                    x_mean_comp[k,i] += w * data[n,i]
            for i in range(dim):
                x_mean_comp[k,i] *= inv_N_comp[k]

    def _update_x_mean_comp(self):
        # (10.52)

        cdef:
            DTYPE_t [:] inv_N_comp = self.inv_N_comp
            DTYPE_t [:,:] r = self.r
            DTYPE_t [:,:] data = self.data
            DTYPE_t [:,:] x_mean_comp = self.x_mean_comp

            cdef size_t K = self.K
            cdef size_t N = len(r)
            cdef size_t dim = self.dim
            cdef size_t k,n,i

        x_mean_comp[:,:] = 0.0

        for k in range(K):
            for n in range(N):
                for i in range(dim):
                    x_mean_comp[k,i] += r[n,k] * data[n,i]
            for i in range(dim):
                x_mean_comp[k,i] *= inv_N_comp[k]

    def _update_S_weighted(self):
        # modified (10.53)

        cdef:
            DTYPE_t w
            DTYPE_t [:] tmpv  = _np.empty_like(self.data[0], dtype=DTYPE)
            DTYPE_t [:] inv_N_comp = self.inv_N_comp
            DTYPE_t [:] weights = self.weights
            DTYPE_t [:,:] data  = self.data
            DTYPE_t [:,:] x_mean_comp = self.x_mean_comp
            DTYPE_t [:,:] r = self.r
            DTYPE_t [:,:,:] S = self.S

            size_t K = self.K
            size_t N = len(r)
            size_t dim = self.dim
            size_t k,n,i,j

        # start with a clean slate
        S[:,:,:] = 0.0

        for k in range(K):
            for n in range(N):
                # copy vector and subtract mean
                for i in range(dim):
                    tmpv[i] = data[n,i] - x_mean_comp[k,i]
                w = weights[n] * r[n,k]
                # outer product on triagonal part
                for i in range(dim):
                    for j in range(i + 1):
                        S[k,i,j] += w * tmpv[i] * tmpv[j]
            # divide by N and restore symmetry
            for i in range(dim):
                for j in range(i + 1):
                    S[k,i,j] *= inv_N_comp[k]
                    S[k,j,i] = S[k,i,j]

        if not _np.isfinite(self.S).any():
            raise _np.linalg.LinAlgError('Encountered inf or nan in update of sample covariance\n' + str(self.S))

    def _update_S(self):
        # (10.53)

        # temp vector and matrix to store outer product
        cdef:
            DTYPE_t [:] tmpv = _np.empty_like(self.data[0], dtype=DTYPE)
            DTYPE_t [:,:,:] S = self.S
            DTYPE_t [:,:] data = self.data
            DTYPE_t [:,:] x_mean_comp = self.x_mean_comp
            DTYPE_t [:,:] r = self.r
            DTYPE_t [:] inv_N_comp = self.inv_N_comp

            size_t K = self.K
            size_t N = len(r)
            size_t dim = self.dim
            size_t k,n,i,j

        # start with a clean slate
        S[:,:,:] = 0.0

        for k in range(K):
            for n in range(N):
                # copy vector and subtract mean
                for i in range(dim):
                    tmpv[i] = data[n,i] - x_mean_comp[k,i]

                # outer product on triagonal part
                for i in range(dim):
                    for j in range(i + 1):
                        S[k,i,j] += r[n,k] * tmpv[i] * tmpv[j]
            # divide by N and restore symmetry
            for i in range(dim):
                for j in range(i + 1):
                    S[k,i,j] *= inv_N_comp[k]
                    S[k,j,i] = S[k,i,j]

        if not _np.isfinite(self.S).any():
            raise _np.linalg.LinAlgError('Encountered inf or nan in update of sample covariance\n' + str(self.S))

    def _update_W(self):
        # (10.62)

        # change order of operations to minimize copying
        for k in range(self.K):
            tmp = self.x_mean_comp[k] - self.m0[k]
            cov = _np.outer(tmp, tmp)
            cov *= self.beta0[k] / (self.beta0[k] + self.N_comp[k])
            cov += self.S[k]
            cov *= self.N_comp[k]
            cov += self.inv_W0[k]
            self.W[k], log_det = chol_inv_det(cov)[1:]
            self.log_det_W[k] = -log_det

    def _update_expectation_log_p_X(self):
        # (10.71)

        self._expectation_log_p_X = 0.
        for k in range(self.K):
            res = 0.
            tmp = self.x_mean_comp[k] - self.m[k]
            res += self.expectation_det_ln_lambda[k]
            res -= self.dim / self.beta[k]
            res -= self.nu[k] * (_np.trace(self.S[k].dot(self.W[k])) + tmp.dot(self.W[k]).dot(tmp))
            res -= self.dim * log(2 * _np.pi)
            res *= self.N_comp[k]
            self._expectation_log_p_X += res

        self._expectation_log_p_X /= 2.0
        return self._expectation_log_p_X

    def _update_expectation_log_p_Z(self):
        # (10.72)

        # simplify to include sum over k: N_k = sum_n r_{nk}

        # contract all indices, no broadcasting
        self._expectation_log_p_Z = _np.einsum('k,k', self.N_comp, self.expectation_ln_pi)
        return self._expectation_log_p_Z

    def _update_expectation_log_p_pi(self):
        # (10.73)

        self._expectation_log_p_pi = Dirichlet_log_C(self.alpha0)
        self._expectation_log_p_pi += _np.einsum('k,k', self.alpha0 - 1, self.expectation_ln_pi)
        return self._expectation_log_p_pi

    def _update_expectation_log_p_mu_lambda(self):
        # (10.74)

        res = 0
        for k in range(self.K):
            tmp = self.m[k] - self.m0[k]
            res += self.dim * log(self.beta0[k] / (2. * _np.pi))
            res += self.expectation_det_ln_lambda[k] - self.dim * self.beta0[k] / self.beta[k] \
                   - self.beta0[k] * self.nu[k] * tmp.dot(self.W[k]).dot(tmp)

            # 2nd part: Wishart normalization
            res +=  2 * Wishart_log_B(self.dim, self.nu0[k], self.log_det_W0[k])

            # 3rd part
            res += (self.nu0[k] - self.dim - 1) * self.expectation_det_ln_lambda[k]

            # 4th part: traces
            res -= self.nu[k] * _np.trace(self.inv_W0[k].dot(self.W[k]))

        self._expectation_log_p_mu_lambda = 0.5 * res
        return self._expectation_log_p_mu_lambda

    def _update_expectation_log_q_Z_weighted(self):
        # modified (10.75)

        self._expectation_log_q_Z = _np.einsum('n,nk,nk', self.weights, self.r, self.log_rho)
        return self._expectation_log_q_Z

    def _update_expectation_log_q_Z(self):
        # (10.75)

        self._expectation_log_q_Z = _np.einsum('nk,nk', self.r, self.log_rho)
        return self._expectation_log_q_Z

    def _update_expectation_log_q_pi(self):
        # (10.76)

        self._expectation_log_q_pi = _np.einsum('k,k', self.alpha - 1, self.expectation_ln_pi) + Dirichlet_log_C(self.alpha)
        return self._expectation_log_q_pi

    def _update_expectation_log_q_mu_lambda(self):
        # (10.77)

        # pull constant out of loop
        res = -0.5 * self.K * self.dim

        for k in range(self.K):
            res += 0.5 * (self.expectation_det_ln_lambda[k] + self.dim * log(self.beta[k] / (2 * _np.pi)))
            # Wishart entropy
            res -= Wishart_H(self.dim, self.nu[k], self.log_det_W[k])

        self._expectation_log_q_mu_lambda = res
        return self._expectation_log_q_mu_lambda

class VBMerge(GaussianInference):
    '''Parsimonious reduction of Gaussian mixture models with a
    variational-Bayes approach [BGP10]_.

    The idea is to reduce the number of components of an overly complex Gaussian
    mixture while retaining an accurate description. The original samples are
    not required, hence it much faster compared to standard variational Bayes.
    The great advantage compared to hierarchical clustering is that the number
    of output components is chosen automatically. One starts with (too) many
    components, updates, and removes those components with vanishing weight
    using  ``prune()``. All the methods the typical user wants to call are taken
    over from and documented in :py:class:`GaussianInference`.

    :param input_mixture:

        MixtureDensity with Gauss components, the input to be compressed.

    :param N:

        The number of (virtual) input samples that the ``input_mixture`` is
        based on. For example, if ``input_mixture`` was fitted to 1000 samples,
        set ``N`` to 1000.

    :param components:

        Integer; the maximum number of output components.

    :param initial_guess:

        MixtureDensity with Gauss components, optional; the starting point
        for the optimization. If provided, its number of components defines
        the maximum possible and the parameter ``components`` is ignored.


    All other keyword arguments are documented in
    :py:meth:`GaussianInference.set_variational_parameters`.

    .. seealso::

        :py:class:`pypmc.density.gauss.Gauss`

        :py:class:`pypmc.density.mixture.MixtureDensity`

        :py:func:`pypmc.density.mixture.create_gaussian_mixture`

    '''

    def __init__(self, input_mixture, N, components=0, initial_guess='first', **kwargs):
        # don't copy input_mixture, we won't update it
        self.input = input_mixture

        # number of input components
        self.L = len(input_mixture.components)

        # input means
        self.mu = _np.array([c.mu for c in self.input.components])

        self._initialize_K(initial_guess, components, kwargs)

        self.dim = len(input_mixture.components[0].mu)

        # need this many responsibilities
        self.N = N

        # effective number of samples per input component
        # in [BGP10], that's N \cdot \omega' (vector!)
        self.Nomega = N * self.input.weights

        self.set_variational_parameters(initial_guess=initial_guess, **kwargs)

        self._initialize_intermediate(self.L)

        # take mean and covariances from initial guess
        if not isinstance(initial_guess, str):
            self._parse_initial_guess(initial_guess)

        self.E_step()

    def _initialize_m(self, initial_guess):
        '''Provide initial guess for ``m`` depending on chosen method in the string ``initial_guess``.'''

        if self.K > self.L:
            raise ValueError("Can't auto-initialize ``m`` with more output components than input components."
                             " Specify ``m`` explicitly.")

        if initial_guess is 'first':
            # the idea behind this is to close to at least one input component
            #  such that components do not die in the very first step only
            # because of a bad initialization
            return _np.array([c.mu for c in self.input.components[:self.K]])
        elif initial_guess is 'random':
            indices = _np.random.choice(len(self.input), size=self.K, replace=False)
            return _np.array([self.input.components[i].mu for i in indices])
        else:
            raise ValueError('Invalid ``initial_guess``: ' + str(initial_guess))

    def _update_expectation_gauss_exponent(self):
        # after (40) in [BGP10]

        cdef:
            double  [:] tmp  = _np.zeros(self.dim, dtype=DTYPE)
            DTYPE_t [:] beta = self.beta
            DTYPE_t [:,:] m  =  self.m
            DTYPE_t [:] nu  =  self.nu
            DTYPE_t [:,:] mu = self.mu
            DTYPE_t [:,:] expectation_gauss_exponent = self.expectation_gauss_exponent
            DTYPE_t [:,:] sigma
            double  [:,:] W
            size_t K = self.K
            size_t L = len(expectation_gauss_exponent)
            size_t dim = self.dim
            size_t k,l,i,j

        for k in range(K):
            W = self.W[k]
            for l in range(L):
                sigma = self.input.components[l].sigma
                for i in range(dim):
                    tmp[i] = mu[l,i] - m[k,i]
                expectation_gauss_exponent[l,k] = dim / beta[k] + nu[k] * bilinear_sym(W, tmp)

    def _update_log_rho(self):
        # (40) in [BGP10]
        # first line: compute k vector
        tmp_k  = 2. * self.expectation_ln_pi
        tmp_k += self.expectation_det_ln_lambda
        tmp_k -= self.dim * _np.log(2. * _np.pi)

        # turn into lk matrix
        self.log_rho = _np.einsum('l,k->lk', self.Nomega, tmp_k)

        # add second line
        self.log_rho -= _np.einsum('l,lk->lk', self.Nomega, self.expectation_gauss_exponent)

        self.log_rho *= 0.5

    def _update_N_comp(self):
        # (41)
        _np.einsum('l,lk', self.Nomega, self.r, out=self.N_comp)
        regularize(self.N_comp)
        self.inv_N_comp = 1. / self.N_comp

    def _update_x_mean_comp(self):
        # (42)
        _np.einsum('k,l,lk,li->ki', self.inv_N_comp, self.Nomega, self.r, self.mu, out=self.x_mean_comp)

    def _update_S(self):
        # combine (43) and (44), since only ever need sum of S and C

        cdef:
            DTYPE_t [:] tmpv  = _np.empty(self.dim, dtype=DTYPE)
            DTYPE_t [:,:,:] S = self.S
            DTYPE_t [:,:] x_mean_comp =  self.x_mean_comp
            DTYPE_t [:,:] mu =  self.mu
            DTYPE_t [:,:] r =  self.r
            DTYPE_t [:,:] sigma
            DTYPE_t [:] inv_N_comp =  self.inv_N_comp
            DTYPE_t [:] Nomega = self.Nomega

            size_t K = self.K
            size_t L = self.L
            size_t dim = self.dim
            size_t k,l,i,j

        # start with a clean slate
        S[:,:,:] = 0.0

        for k in range(K):
            for l in range(L):
                # copy vector and subtract mean
                for i in range(dim):
                    tmpv[i] = mu[l,i] - x_mean_comp[k,i]

                sigma = self.input.components[l].sigma

                # outer product on triagonal part
                for i in range(dim):
                    for j in range(i + 1):
                        S[k,i,j] += Nomega[l] * r[l,k] * (tmpv[i] * tmpv[j] + sigma[i,j])
            # divide by N_comp and restore symmetry
            for i in range(dim):
                for j in range(i + 1):
                    S[k,i,j] *= inv_N_comp[k]
                    S[k,j,i] = S[k,i,j]

def Wishart_log_B(D, nu, log_det):
    '''Compute first part of a Wishart distribution's normalization,
    (B.79) of [Bis06]_, on the log scale.

    :param D:

        Dimension of parameter vector; i.e. ``W`` is a DxD matrix.

    :param nu:

        Degrees of freedom of a Wishart distribution.

    :param log_det:

        The determinant of ``W``, :math:`|W|`.

    '''
    assert D > 0, 'Invalid dimension: %s' % D
    assert nu > D - 1, 'Invalid degree of freedom: %s' % nu
    assert _np.isfinite(log_det), 'Non-finite log(det): %s' % log_det

    log_B = -0.5 * nu * log_det
    log_B -= 0.5 * nu * D * log(2)
    log_B -= 0.25 * D * (D - 1) * log(_np.pi)
    for i in range(1, D + 1):
        log_B -= _gammaln(0.5 * (nu + 1 - i))

    return log_B

def Wishart_expect_log_lambda(D, nu, log_det):
    ''' :math:`E[\log |\Lambda|]`, (B.81) of [Bis06]_ .'''
    assert D > 0, 'Invalid dimension: %s' % D
    assert nu > D - 1, 'Invalid degree of freedom: %s' % nu
    assert _np.isfinite(log_det), 'Non-finite log(det): %s' % log_det

    result = 0
    for i in range(1, D + 1):
        result += _digamma(0.5 * (nu + 1 - i))
    return result + D * log(2.) + log_det

def Wishart_H(D, nu, log_det):
    '''Entropy of the Wishart distribution, (B.82) of [Bis06]_ .'''

    log_B = Wishart_log_B(D, nu, log_det)

    expect_log_lambda = Wishart_expect_log_lambda(D, nu, log_det)

    return -log_B - 0.5 * (nu - D - 1) * expect_log_lambda + 0.5 * nu * D

def Dirichlet_log_C(alpha):
    '''Compute normalization constant of Dirichlet distribution on
    log scale, (B.23) of [Bis06]_ .

    '''

    # compute gamma functions on log scale to avoid overflows
    log_C = _gammaln(_np.einsum('k->', alpha))
    for alpha_k in alpha:
        log_C -= _gammaln(alpha_k)

    return log_C
