"""Variational clustering as described in [Bis06]_

"""

from __future__ import division
from .gaussian_mixture import GaussianMixture
from math import log
import numpy as _np
from scipy.special import gamma as _gamma
from scipy.special import gammaln as _gammaln
from scipy.special.basic import digamma as _digamma
from .._tools._doc import _inherit_docstring, _add_to_docstring

def regularize(x):
    '''Replace zeros by smallest positive float.

    :param x:

        Numpy-array

    Return x

    '''
    x[_np.where(x == 0)] = _np.finfo('d').tiny
    return x

class _Inference(object):
    '''Abstract base class; approximates a probability density by a
    member of a specific class of probability densities

    '''

    def __init__(self):
        raise NotImplementedError('Do not create instances from this class, use derived classes instead.')

    def get_result(self):
        '''Returns the parameters calculated by ``self.update``'''
        raise NotImplementedError()

    def likelihood_bound(self):
        '''Compute the lower bound on the true log marginal likelihood
        :math:`L(Q)` given the current parameter estimates.

        '''
        raise NotImplementedError()

    def prune(self, threshold=1.):
        '''Delete components with an effective number of samples
        :math:`N_k` below the threshold.

        :param threshold:

            Float; the minimum effective number of samples a component must have
            to survive.

        '''
        raise NotImplementedError()

    def set_variational_parameters(self, *args, **kwargs):
        '''Reset the parameters to the submitted values or default
        Use this function to set initial values for the iteration.

        '''
        raise NotImplementedError()

    def update(self):
        '''Recalculate the parameters (M step) and expectation values (E step)
        using the update equations.

        '''
        raise NotImplementedError()

class GaussianInference(_Inference):
    '''Approximate a probability density by a Gaussian mixture with a variational
    Bayes approach. The motivation, notation, and derivation is explained in
    detail in chapter 10.2 in [Bis06]_.

    .. seealso ::

        Another implementation can be found at https://github.com/jamesmcinerney/vbmm.


    :param data:

        Matrix like array; Each of the :math:`N` rows contains one
        :math:`D`-dimensional sample from the probability density to be
        approximated.

    :param components:

        Integer; :math:`K` is the number of Gaussian components in the
        approximating Gaussian mixture.

    All keyword arguments are processed by :py:meth:`set_variational_parameters`.

    '''
    def __init__(self, data, components, **kwargs):
        self.data = data
        self.components = components
        self.N, self.dim = self.data.shape

        self.set_variational_parameters(**kwargs)

        # compute expectation values for the initial parameter values
        # so a valid bound can be computed after object is initialized
        self.E_step()

    @_add_to_docstring(
        r'''

        The prior (and posterior) probability distribution of :math:`\boldsymbol{\mu}` and
        :math:`\boldsymbol{\Lambda}` is given by

        .. math::

            p(\boldsymbol{\mu}, \boldsymbol{\Lambda}) =
            p(\boldsymbol{\mu}|\boldsymbol{\Lambda}) p(\boldsymbol{\Lambda}) =
            \prod_{k=1}^K
              \mathcal{N}(\boldsymbol{\mu}_k|\boldsymbol{m_k},(\beta_k\boldsymbol{\Lambda}_k)^{-1})
              \mathcal{W}(\boldsymbol{\Lambda}_k|\boldsymbol{W_k}, \nu_k),

        where :math:`\mathcal{N}` denotes a Gaussian and :math:`\mathcal{W}`
        a Wishart distribution.

        .. warning ::

            This function may delete results obtained by ``self.update``.

        .. note::

            For good performance, it is strongly recommended to explicitly
            initialize ``m`` to values close to the bulk of the target
            distribution. For all other parameters, consult chapter 10
            in [Bis06]_ when considering to modify the defaults.

        :param alpha0:

            Float; :math:`\alpha_0` parameter for the mixing coefficients'
            probability distribution

            .. math::
                p(\boldsymbol{\pi}) = Dir(\boldsymbol{\pi}|\boldsymbol{\alpha_0})

            where Dir denotes the Dirichlet distribution and

            .. math::
                \boldsymbol{\alpha_0} = (\alpha_0,\dots,\alpha_0).

            Default:

            .. math::
                \alpha_0 = 10^{-5}.

        :param beta0:

            Float; :math:`\beta_0` parameter of the probability distribution of
            :math:`\boldsymbol{\mu}` and :math:`\boldsymbol{\Lambda}`. Should be
            of the same order as ``alpha0`` for numerical stability. Default:

            .. math::
                \beta_0 = 10^{-5}.

        :param nu0:

            Float; :math:`\nu_0` is the minimum value of the number of degrees of
            freedom of the Wishart distribution of :math:`\boldsymbol{\Lambda}`.
            It is `not` updated and common to all components. To avoid a
            divergence at the origin, it is required that

            .. math::
                \nu_0 \geq D + 1 .

            Default:

            .. math::
                \nu_0 = D+1.

        :param nu:

            :math:`K` vector; The number of degrees of freedom of the Wishart
            distribution of the precision matrix of each output component. The
            default initialization is

            .. math::
                \nu_k = \nu_0 + N / K
            in anticipation of the updating rule for :math:`\nu_k`.

        :param m0:

            :math:`K` vector; regularization parameter for the Gausian means.
            Default:

            .. math::
                \boldsymbol{m_0} = (0,\dots,0)

        :param m:

            :math:`(K \times D)` matrix-like array; :math:`\boldsymbol{m}_k` is
            the mean parameter of the Gaussian distribution of the :math:`k`-th
            component mean :math:`\boldsymbol{\mu}_k`. Default: The sequence of
            :math:`K \times D` equally spaced values in [-1,1] reshaped to
            :math:`K \times D` dimensions.

            .. warning:: If component means are identical initially, they
                may remain identical. It is advisable to randomly scatter
                them in order to avoid singular behavior.

        :param W0:

            :math:`D \times D` matrix-like array; :math:`\boldsymbol{W_0}` is a
            symmetric positive-definite matrix taken as the regularization
            parameter `and` initial value for updates of the Wishart parameter
            :math:`W_k`. Default: identity matrix.

        ''')
    @_inherit_docstring(_Inference)
    def set_variational_parameters(self, *args, **kwargs):
        if args: raise TypeError('keyword args only')

        alpha0 = kwargs.pop('alpha0', 1e-5)
        self.alpha0 = _np.ones(self.components) * alpha0

        # in the limit beta --> 0: uniform prior
        self.beta0  = kwargs.pop('beta0' , 1e-5)

        # smallest possible nu such that the Wishart pdf does not diverge at 0
        self.nu0    = kwargs.pop('nu0'   , self.dim + 1.)
        if self.nu0 < self.dim + 1.:
            raise ValueError('nu0 (%g) must exceed %g to avoid divergence at the origin.' % (self.nu0, self.dim + 1.))
        self.nu     = kwargs.pop('nu'   , _np.zeros(self.components) + self.nu0)

        self.m0     = kwargs.pop('m0'    , _np.zeros(self.dim))
        self.m      = kwargs.pop('m'     , None)
        if self.m is None:
            # TODO: maybe remove standard value because this won't perform well on components far away from zero
            # If the initial means are identical, the components remain identical in all updates.
            self.m = _np.linspace(-1.,1., self.components*self.dim).reshape((self.components, self.dim))

        # covariance matrix; unit matrix <--> unknown correlation
        self.W0     = kwargs.pop('W0'    , None)
        if self.W0 is None:
            self.W0     = _np.eye(self.dim)
            self.inv_W0 = self.W0.copy()
        else:
            self.inv_W0 = _np.linalg.inv(self.W0)

        if kwargs: raise TypeError('unexpected keyword(s): ' + str(kwargs.keys()))

        self._initialize_output()

    def _initialize_output(self):
        '''Create all variables needed for the iteration in ``self.update``'''
        self.x_mean_comp = _np.zeros((self.components, self.dim))
        self.W = _np.array([self.W0 for i in range(self.components)])
        self.expectation_gauss_exponent = _np.zeros((  self.N,self.components  ))
        self.N_comp = self.N / self.components * _np.ones(self.components)
        self.alpha = _np.array(self.alpha0)
        self.beta  = self.beta0  * _np.ones(self.components)
        self.S = _np.empty_like(self.W)

    #TODO: move citation to function itself

    def E_step(self):
        self._update_expectation_gauss_exponent() #eqn 10.64 in [Bis06]
        self._update_expectation_det_ln_lambda() #eqn 10.65 in [Bis06]
        self._update_expectation_ln_pi() #eqn 10.66 in [Bis06]
        self._update_r() #eqn 10.46 and 10.49 in [Bis06]
        self._update_N_comp() #eqn 10.51 in [Bis06]
        self._update_x_mean_comp() #eqn 10.52 in [Bis06]
        self._update_S() #eqn 10.53 in [Bis06]

    def M_step(self):
        self.nu = self.nu0 + self.N_comp #eqn 10.63 in [Bis06]
        self.alpha = self.alpha0 + self.N_comp #eqn 10.58 in [Bis06]
        self.beta = self.beta0 + self.N_comp #eqn 10.60 in [Bis06]
        self._update_m() #eqn 10.61 in [Bis06]
        self._update_W() #eqn 10.62 in [Bis06]

    @_inherit_docstring(_Inference)
    def update(self):
        self.M_step()
        self.E_step()

        # TODO: implement support for weights

    def _update_log_rho(self):
        # todo check sum of vector and matrix
        self.log_rho = self.expectation_ln_pi + 0.5 * self.expectation_det_ln_lambda \
                  - 0.5 * self.dim * log(2. * _np.pi) - 0.5 * self.expectation_gauss_exponent

    def _update_m(self):
        for k in range(self.components):
            self.m[k] = 1. / self.beta[k] * (self.beta0 * self.m0 + self.N_comp[k] * self.x_mean_comp[k])

    def _update_N_comp(self):
        self.N_comp = _np.einsum('nk->k', self.r)
        self.inv_N_comp = 1. / regularize(self.N_comp)

    def _update_r(self):
        self._update_log_rho()

        # rescale log to avoid division by zero:
        # find largest log for fixed comp. k
        # and subtract it s.t. largest value at 0 (or 1 on linear scale)
        rho = self.log_rho - self.log_rho.max(axis=1).reshape((len(self.log_rho), 1))
        rho = _np.exp(rho)

        # compute normalization for each comp. k
        normalization_rho = rho.sum(axis=1).reshape((len(rho), 1))

        # in the division, the extra scale factor drops out automagically
        self.r = rho / normalization_rho

        # avoid overflows and nans when taking the log of 0
        regularize(self.r)

    def _update_expectation_det_ln_lambda(self):
        # negative determinants from improper matrices trigger ValueError on some machines only;
        # so test explicitly
        dets = _np.array([_np.linalg.det(W) for W in self.W])
        assert (dets > 0).all(), 'Some precision matrix is not positive definite in %s' % self.W

        res = _np.zeros_like(self.nu)
        tmp = _np.zeros_like(self.nu)
        for i in range(1, self.dim + 1):
            tmp[:] = self.nu
            tmp += 1. - i
            tmp *= 0.5
            # digamma aware of vector input
            res += _digamma(tmp)

        res += self.dim * log(2.)
        res += _np.log(dets)

        self.expectation_det_ln_lambda = res

    def _update_expectation_gauss_exponent(self): #_expectation_gauss_exponent --> _expectation_gauss_exponent[n,k]
        for k in range(self.components):
            for n in range(self.N):
                tmp                                  = self.data[n] - self.m[k]
                self.expectation_gauss_exponent[n,k] = self.dim / self.beta[k] + self.nu[k] * tmp.dot(self.W[k]).dot(tmp)

    def _update_expectation_ln_pi(self):
        self.expectation_ln_pi = _digamma(self.alpha) - _digamma(self.alpha.sum())

    def _update_x_mean_comp(self):
        _np.einsum('k,nk,ni->ki', self.inv_N_comp, self.r, self.data, out=self.x_mean_comp)

    def _update_S(self):
        # eqn (10.53) in [Bis06]
        # use outer product to guarantee a positive definite symmetric S
        # expanding it into four terms, then using einsum failed numerically for large N
        for k in range(self.components):
            self.S[k,:,:] = 0
            for n, x in enumerate(self.data):
                tmp = x - self.x_mean_comp[k]
                self.S[k] += self.r[n,k] * _np.outer(tmp, tmp)
            self.S[k] *= self.inv_N_comp[k]

    def _update_W(self):
        # (10.62)
        # change order of operations to minimize copying
        for k in range(self.components):
            tmp = self.x_mean_comp[k] - self.m0
            cov = _np.outer(tmp, tmp)
            cov *= self.beta0 / (self.beta0 + self.N_comp[k])
            cov += self.S[k]
            cov *= self.N_comp[k]
            cov += self.inv_W0
            self.W[k] = _np.linalg.inv(cov)

    def get_result(self):
        '''Return the mixture-density computed from the
        mode of the variational-Bayes estimate.

        '''

        # find mode of Gaussian-Wishart distribution
        # and invert to find covariance. The result
        # \Lambda_k = (\nu_k - D) W_k
        # turns out to be independent of beta.

        # The most likely value of the mean is m_k,
        # the mean parameter of the Gaussian q(\mu_k).

        # The mode of the Dirichlet exists only if \alpha_k > 1.
        components = []
        weights = []
        for k, W in enumerate(self.W):
            if self.nu[k] >= self.dim + 1:
                W = (self.nu[k] - self.dim) * W
                cov = _np.linalg.inv(W)
                components.append(GaussianMixture.Component(self.m[k], cov))
                # copy over precision matrix
                components[-1].inv = W

                # Dirichlet mode
                pi = (self.alpha[k] - 1.) / (self.alpha.sum() - self.components)
                assert pi >= 0, 'Mode of Dirichlet distribution requires alpha_k > 1 ' + \
                                '(alpha_k=%g) and at least K > 2 (K=%g)' % (self.alpha[k], self.components)

                # relative weight properly normalized
                weights.append(pi)
            else:
                print('Skipping comp. %d with dof %g' % (k,self.nu[k]))

        return GaussianMixture(components, weights)

    @_inherit_docstring(_Inference)
    def prune(self, threshold = 1.):

        components_to_survive = _np.where(self.N_comp >= threshold)[0]
        self.components = len(components_to_survive)

        # list all vector and matrix vmembers
        vmembers = ('alpha0', 'alpha', 'beta', 'expectation_det_ln_lambda',
                   'expectation_ln_pi', 'N_comp', 'nu', 'm', 'S', 'W', 'x_mean_comp')
        mmembers = ('expectation_gauss_exponent', 'r')

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
            setattr(self, m, getattr(self, m)[:self.components])
        for m in mmembers:
            setattr(self, m, getattr(self, m)[:, :self.components])

    def likelihood_bound(self):
        # todo easy to parallize sum of independent terms
        bound  = self._expect_log_p_X() # (10.71)
        bound += self._expect_log_p_Z() # (10.72)
        bound += self._expect_log_p_pi() # (10.73)
        bound += self._expect_log_p_mu_lambda() # (10.74)
        bound -= self._expect_log_q_Z() # (10.75)
        bound -= self._expect_log_q_pi() # (10.76)
        bound -= self._expect_log_q_mu_lambda() # (10.77)

        return bound

    def _expect_log_p_X(self):
        # (10.71)

        result = 0
        for k in range(self.components):
            tmp = self.x_mean_comp[k] - self.m[k]
            result += self.N_comp[k] * \
                      (self.expectation_det_ln_lambda[k] - self.dim / self.beta[k]
                       -self.nu[k] * (_np.trace(self.S[k].dot(self.W[k]))
                       + tmp.dot(self.W[k]).dot(tmp)) - self.dim * log(2 * _np.pi))

        return 0.5 * result

    def _expect_log_p_Z(self):
        # simplify (10.72) to include sum over k
        # N_k = sum_n r_{nk}

        # contract all indices, no broadcasting
        return _np.einsum('k,k', self.N_comp, self.expectation_ln_pi)

    def _expect_log_p_pi(self):
        # (10.73)

        log_C = Dirichlet_log_C(self.alpha0)

        return log_C + _np.einsum('k,k', self.alpha0 - 1, self.expectation_ln_pi)

    def _expect_log_p_mu_lambda(self):
        # (10.74)

        result = 0
        for k in range(self.components):
            tmp = self.m[k] - self.m0
            result += self.expectation_det_ln_lambda[k] - self.dim * self.beta0 / self.beta[k] \
                      - self.beta0 * self.nu[k] * tmp.dot(self.W[k]).dot(tmp)

        result *= 0.5
        result += self.components * 0.5 * self.dim * log(self.beta0 / (2. * _np.pi))

        # compute Wishart normalization
        result += self.components * Wishart_log_B(self.W0, self.nu0)

        # third part
        result += 0.5 * (self.nu0 - self.dim - 1) * _np.einsum('k->', self.expectation_det_ln_lambda)

        # final part
        traces = 0
        for nu_k, W_k in zip(self.nu, self.W):
            traces += nu_k * _np.trace(self.inv_W0.dot(W_k))
        result -= 0.5 * traces

        return result

    def _expect_log_q_Z(self):
        # (10.75)

        return _np.einsum('nk,nk', self.r, _np.log(self.r))

    def _expect_log_q_pi(self):
        # (10.76)

        return _np.einsum('k,k', self.alpha - 1, self.expectation_ln_pi) + Dirichlet_log_C(self.alpha)

    def _expect_log_q_mu_lambda(self):
        # (10.77)

        # pull constant out of loop
        result = -0.5 * self.components * self.dim

        for k in range(self.components):
            result += 0.5 * (self.expectation_det_ln_lambda[k] + self.dim * log(self.beta[k] / (2 * _np.pi)))
            # Wishart entropy
            result -= Wishart_H(self.W[k], self.nu[k])

        return result

class VBMerge(GaussianInference):
    # todo refer to set_variational_parameters for more kwargs
    '''Parsimonious reduction of Gaussian mixture models with a
    variational-Bayes approach [BGP10]_.

    The idea is to reduce the number of components of an overly complex Gaussian
    mixture while retaining an accurate description. The original samples are
    not required, hence it much faster compared to standard variational Bayes.
    The great advantage compared to hierarchical clustering is that the number
    of output components is chosen automatically. One starts with (too) many
    components, updates, and removes those components with vanishing weight
    using  ``prune()``.

    :param input_mixture:

        A Gaussian mixture density, the input to be compressed.

    :param initial_guess:

        A Gaussian mixture density, the starting point for the optimization.
        Its number of components defines the maximum possible.

    :param N:

        The number of (virtual) input samples that the ``input_mixture`` is
        based on. For example, if ``input_mixture`` was fitted to 1000 samples,
        set ``N`` to 1000.

    .. seealso::
        :py:class:`.gaussian_mixture.GaussianMixture`

    '''

    def __init__(self, input_mixture, N, components=None, initial_guess=None, **kwargs):
        # make sure input has the correct inverse matrices available
        self.input = input_mixture
        for c in self.input:
            c._inv()

        # number of input components
        self.L = len(input_mixture.comp)

        # input means
        self.mu = _np.array([c.mean for c in self.input])

        if initial_guess is not None:
            self.components = len(initial_guess.comp)
        elif components is not None:
            self.components = components
        else:
            raise ValueError('Specify either `components` or `initial_guess` to set the initial values')

        self.dim = len(input_mixture[0].mean)
        self.N = N

        # effective number of samples per input component
        # in [BGP10], that's N \cdot \omega' (vector!)
        self.Nomega = self.input.w * N

        self.set_variational_parameters(**kwargs)
        # take mean and covariances from initial guess
        if initial_guess is not None:
            # precision matrix is inverse of covariance
            for c in initial_guess:
                c._inv()
            self.W = _np.array([c.inv for c in initial_guess])

            # copy over the means
            self.m = _np.array([c.mean for c in initial_guess])
            #TODO: copy weights

        self.E_step()

    def _initialize_output(self):
        GaussianInference._initialize_output(self)
        self.expectation_gauss_exponent = _np.zeros((self.L, self.components))

    # E- and M step almost as in standard VB, differences implemented in
    # overloaded functions, the equation numbers are:

#     def E_step(self):
#         self._update_expectation_det_ln_lambda() # (21)
#         self._update_expectation_ln_pi() # (22)
#         self._update_expectation_gauss_exponent() # after (40)
#         self._update_r() # after (40)
#         self._update_N_comp() # (41)
#         # synthetic statistics
#         self._update_x_mean_comp() # (42)
#         self._update_S() # (43) and (44)

#     def M_step(self):
#         self.alpha = self.alpha0 + self.N_comp # (45)
#         self.beta = self.beta0 + self.N_comp # (46)
#         self._update_m() # (47)
#         self._update_W() # (48)
#         self.nu = self.nu0 + self.N_comp # (49)

    def _update_expectation_gauss_exponent(self):
        for k, W in enumerate(self.W):
            for l, comp in enumerate(self.input):
                tmp = comp.mean - self.m[k]
                self.expectation_gauss_exponent[l,k] = self.dim / self.beta[k] + self.nu[k] * \
                                                       (_np.trace(W.dot(comp.inv)) + tmp.dot(W).dot(tmp))

    def _update_log_rho(self):
        # eqn 40 in [BGP10]
        # first line: compute k vector
        tmp_k  = 2 * self.expectation_ln_pi
        tmp_k += self.expectation_det_ln_lambda
        tmp_k -= self.dim * _np.log(2 * _np.pi)

        # turn into lk matrix
        self.log_rho = _np.einsum('l,k->lk', self.Nomega, tmp_k)

        # add second line
        self.log_rho -= _np.einsum('l,lk->lk', self.Nomega, self.expectation_gauss_exponent)

        self.log_rho /= 2.0

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

        for k in range(self.components):
            self.S[k,:] = 0.0
            for l in range(self.L):
                tmp        = self.mu[l] - self.x_mean_comp[k]
                self.S[k] += self.Nomega[l] * self.r[l,k] * (_np.outer(tmp, tmp) + self.input[l].cov)

            self.S[k] *= self.inv_N_comp[k]

    def _expect_log_q_Z(self):
        # missing in [BGP10]
        # simplify (10.75) of [Bis06] assuming that all samples from the
        # l-th component have identical r_{nk}
        # E[log(q(Z))] = N \omega'_l r_{lk} \log r_{lk}

        return _np.einsum('l,lk,lk', self.Nomega, self.r, _np.log(self.r))

# todo move Wishart stuff to separate class, file?
# todo doesn't check that nu > D - 1
def Wishart_log_B(W, nu, det=None):
    '''Compute first part of a Wishart distribution's normalization,
    (B.79) of [Bis06]_, on the log scale.

    :param W:

        Covariance matrix of a Wishart distribution.

    :param nu:

        Degrees of freedom of a Wishart distribution.

    :param det:

        The determinant of ``W``, :math:`|W|`. If `None`, recompute it.

    '''

    if det is None:
        det = _np.linalg.det(W)

    log_B = -0.5 * nu * log(det)
    log_B -= 0.5 * nu * len(W) * log(2)
    log_B -= 0.25 * len(W) * (len(W) - 1) * log(_np.pi)
    for i in range(1, len(W) + 1):
        log_B -= _gammaln(0.5 * (nu + 1 - i))

    return log_B

def Wishart_expect_log_lambda(W, nu):
    ''' E[log |\Lambda|], (B.81) of [Bis06]_ .'''
    result = 0
    for i in range(1, len(W) + 1):
        result += _digamma(0.5 * (nu + 1 - i))
    result += len(W) * log(2.)
    result += log(_np.linalg.det(W))
    return result

def Wishart_H(W, nu):
    '''Entropy of the Wishart distribution, (B.82) of [Bis06]_ .'''

    log_B = Wishart_log_B(W, nu)

    expect_log_lambda = Wishart_expect_log_lambda(W, nu)

    # dimension
    D = len(W)

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
