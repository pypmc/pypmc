"""Variational clustering as described in [Bis06]_

"""

from __future__ import division as _div
from .gaussian_mixture import GaussianMixture
from math import log
import numpy as _np
from scipy.special import gamma as _gamma
from scipy.special import gammaln as _gammaln
from scipy.special.basic import digamma as _digamma
from .._tools._doc import _inherit_docstring, _add_to_docstring

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
        '''Deletes components with small effective number of samples :math:`N_k`

        :param threshold:

            Float; the minimum effective number of samples :math:`N_k`
            a component must have to survive.

        '''
        raise NotImplementedError()

    def set_variational_parameters(self, *args, **kwargs):
        '''Resets the parameters to the submitted values or default
        Use this function to set initial values for the iteration.

        '''
        raise NotImplementedError()

    def update(self, N=1):
        '''Recalculates the parameters using the update equations

        :param N:

            An int which defines the maximum number of steps to run the
            iteration.

        '''
        raise NotImplementedError()

class GaussianInference(_Inference):
    # todo refer to set_variational_parameters for more kwargs
    '''Approximates a probability density by a Gaussian mixture according
    to chapter 10.2 in [Bis06]_

    .. seealso ::

        Another implementation can be found at https://github.com/jamesmcinerney/vbmm


    :param data:

        Matrix like array; random points drawn from the probability density
        to be approximated. One row represents one n-dim point.

    :param components:

        Integer, the number of Gaussian components in the approximating
        multimodal Gaussian

    '''
    def __init__(self, data, components, **kwargs):
        # todo but data is not changed, so why a copy?
        self.data       = _np.array(data) #call array constructor to be sure to have a copy
        self.components = components
        self.N, self.dim = self.data.shape

        self.set_variational_parameters(**kwargs)

    @_add_to_docstring(
        r'''.. warning ::

            This function may delete results obtained by ``self.update``.

        .. note::

            Before using other parameter values than default for initialization,
            you are strongly recommended to read chapter 10 in [Bis06]_.


        :param alpha0:

            Float; :math:`\alpha_0` parameter for the mixing coefficients'
            probability distribution

            .. math::
                p(\boldsymbol{\pi}) = Dir(\boldsymbol{\pi}|\boldsymbol{\alpha_0})

            where Dir denotes the Dirichlet distribution and

            .. math::
                \boldsymbol{\alpha_0} = (\alpha_0,...,\alpha_0)

        :param beta0:

            Float; :math:`\beta_0` parameter of the probability distribution of
            :math:`\boldsymbol{\mu}` and :math:`\boldsymbol{\Lambda}`. Should be
            of the same order as ``alpha0`` for numerical stability.

        :param nu0:

            Float; :math:`\nu_0` parameter for the probability distribution of
            :math:`\boldsymbol{\mu}` and :math:`\boldsymbol{\Lambda}`.

        :param m0:

            Matrix like array; :math:`m_0` parameter for the probability
            distribution of
            :math:`\boldsymbol{\mu}` and :math:`\boldsymbol{\Lambda}`.
            Provide an initial guess for the means such that the
            mean of component i can be accessed as m0[i].

            @warning: If component means are identical initially, they
                may remain identical. It is advisable to randomly initialize
                them in order to avoid such singular behavior.

        :param W0:

            Matrix like array; :math:`\boldsymbol{W_0}` parameter for the
            probability distribution of
            :math:`\boldsymbol{\mu}` and :math:`\boldsymbol{\Lambda}`.


        The prior probability distribution of :math:`\boldsymbol{\mu}` and
        :math:`\boldsymbol{\Lambda}` is given by

        .. math::

            p(\boldsymbol{\mu}, \boldsymbol{\Lambda}) =
            p(\boldsymbol{\mu}|\boldsymbol{\Lambda}) p(\boldsymbol{\Lambda}) =
            \prod_{k=1}^K
              \mathcal{N}(\boldsymbol{\mu}_k|\boldsymbol{m_0},(\beta_0\boldsymbol{\Lambda}_k)^{-1})
              \mathcal{W}(\boldsymbol{\Lambda}_k|\boldsymbol{W_0}, \nu_0)

        where :math:`\mathcal{N}` denotes a Gaussian and :math:`\mathcal{W}`
        a Wishart distribution.

        ''')
    @_inherit_docstring(_Inference)
    def set_variational_parameters(self, *args, **kwargs): # TODO: write default initial values into docstring
        if args != (): raise TypeError('keyword args only; try set_adapt_parameters(keyword = value)')

        self.alpha0 = kwargs.pop('alpha0', 1e-5)

        # in the limit beta --> 0: uniform prior
        self.beta0  = kwargs.pop('beta0' , 1e-5)

        # smallest possible nu such that the Wishart pdf does not diverge at 0
        self.nu0    = kwargs.pop('nu0'   , self.dim + 1.)

        self.m0     = kwargs.pop('m0'    , _np.zeros(self.dim))
        self.m      = kwargs.pop('m'     , None)
        if self.m is None:
            # If the initial means are identical, the components remain identical in all updates.
            self.m = _np.linspace(-1.,1., self.components*self.dim).reshape((self.components, self.dim))

        # covariance matrix; unit matrix <--> unknown correlation
        self.W0     = kwargs.pop('W0'    , None)
        if self.W0 is None:
            self.W0     = _np.eye(self.dim)
            self.inv_W0 = self.W0
        else:
            self.inv_W0 = _np.linalg.inv(self.W0)

        if kwargs: raise TypeError('unexpected keyword(s): ' + str(kwargs.keys()))

        self._initialize_output()

    def _initialize_output(self):
        '''Create all variables needed for the iteration in ``self.update``'''
        self.x_mean_comp = _np.zeros((self.components, self.dim))
        self.nu = _np.zeros(self.components) + self.nu0 # todo was + 1 before. Why?
        self.W = _np.array([self.W0 for i in range(self.components)])
        self.expectation_gauss_exponent = _np.zeros((  self.N,self.components  ))
        self.N_comp = self.N / self.components * _np.ones(self.components)
        self.alpha = self.alpha0 * _np.ones(self.components)
        self.beta  = self.beta0  * _np.ones(self.components)
        self.S = _np.empty_like(self.W)

    # ------------------- below belongs to update ---------------------
    @_inherit_docstring(_Inference)
    def update(self, N=1):
        for i in range(N):
            # E-like step
            self._update_expectation_gauss_exponent() #eqn 10.64 in [Bis06]
            self._update_expectation_det_ln_lambda() #eqn 10.65 in [Bis06]
            self._update_expectation_ln_pi() #eqn 10.66 in [Bis06]
            self._update_r() #eqn 10.46 and 10.49 in [Bis06]

            # M-like step
            self.N_comp = self.r.sum(axis=0) #eqn 10.51 in [Bis06]
            self.nu = self.nu0 + self.N_comp #eqn 10.63 in [Bis06]
            self._update_x_mean_comp() #eqn 10.52 in [Bis06]
            self._update_S() #eqn 10.53 in [Bis06]
            self.alpha = self.alpha0 + self.N_comp #eqn 10.58 in [Bis06]
            self.beta = self.beta0 + self.N_comp #eqn 10.60 in [Bis06]
            self._update_m() #eqn 10.61 in [Bis06]
            self._update_W() #eqn 10.62 in [Bis06]

            # TODO: insert convergence criterion --> lower log-likelihood bound
            # TODO: implement support for weights

    def _update_m(self):
        for k in range(self.components):
            self.m[k] = 1./self.beta[k] * (self.beta0*self.m0[k] + self.N_comp[k]*self.x_mean_comp[k])

    def _update_log_rho(self):
        # todo check sum of vector and matrix
        self.log_rho = self.expectation_ln_pi + 0.5 * self.expectation_det_ln_lambda \
                  - 0.5 * self.dim * log(2. * _np.pi) - 0.5 * self.expectation_gauss_exponent

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

    def _update_expectation_det_ln_lambda(self):
        logdet_W = _np.array([log(_np.linalg.det(self.W[k])) for k in range(self.components)])

        tmp = 0.
        for i in range(1, self.dim + 1):
            tmp += _digamma(    0.5 * (self.nu + 1. - i)    )

        self.expectation_det_ln_lambda = tmp + self.dim * log(2.) + logdet_W

    def _update_expectation_gauss_exponent(self): #_expectation_gauss_exponent --> _expectation_gauss_exponent[n,k]
        for k in range(self.components):
            for n in range(self.N):
                tmp                                  = self.data[n] - self.x_mean_comp[k]
                self.expectation_gauss_exponent[n,k] = self.dim / self.beta[k] + self.nu[k] * tmp.transpose().dot(self.W[k]).dot(tmp)

    def _update_expectation_ln_pi(self):
        self.expectation_ln_pi = _digamma(self.alpha) - _digamma(self.alpha.sum())

    def _update_x_mean_comp(self):
        # todo use np.average
        for k in range(self.components):
            if not self.N_comp[k] == 0: # prevent errors and x_mean is unimportant for a dead component
                self.x_mean_comp[k] = 1./self.N_comp[k] * (self.r[:,k] * self.data.T).T.sum(axis = 0)

    def _update_S(self):
        self.S = _np.zeros_like(self.S)
        for k in range(self.components):
            for n in range(self.N):
                if not self.N_comp[k] == 0: # prevent errors and S for a dead component is unimportant
                    tmp        = _np.array([self.data[n] - self.x_mean_comp[k]])
                    self.S[k] += 1./self.N_comp[k] * self.r[n,k] * _np.outer(tmp, tmp)

    def _update_W(self):
        for k in range(self.components):
            tmp = _np.array([self.x_mean_comp[k] - self.m0[k]])
            cov = self.inv_W0 + self.N_comp[k] * self.S[k]
            self.W[k] = _np.linalg.inv(cov)

    # ------------------- above belongs to update ---------------------

    def get_result(self):
        '''Returns the parameters calculated by ``self.update`` as
        tuple(abundances, means, covariances)

        '''
        return self.N_comp/self.N,self.m,self.S

    @_inherit_docstring(_Inference)
    def prune(self, threshold = 1.):
        components_to_survive = _np.where(self.N_comp >= threshold)[0]
        self.components = len(components_to_survive)

        k_new = 0
        for k_old in components_to_survive:
            # reindex surviving components
            if k_old != k_new:
                self.expectation_gauss_exponent[:,k_new] = self.expectation_gauss_exponent[:,k_old]
                self.expectation_det_ln_lambda   [k_new] = self.expectation_det_ln_lambda   [k_old]
                self.alpha                       [k_new] = self.alpha                       [k_old]
                self.expectation_ln_pi           [k_new] = self.expectation_ln_pi           [k_old]
                self.r                         [:,k_new] = self.r                         [:,k_old]

                self.N_comp                      [k_new] = self.N_comp                      [k_old]
                self.nu                          [k_new] = self.nu                          [k_old]
                self.x_mean_comp                 [k_new] = self.x_mean_comp                 [k_old]
                self.S                           [k_new] = self.S                           [k_old]
                self.beta                        [k_new] = self.beta                        [k_old]
                self.m                           [k_new] = self.m                           [k_old]
                self.W                           [k_new] = self.W                           [k_old]
            k_new += 1

        # cut the unneccessary part of the data
        self.expectation_gauss_exponent = self.expectation_gauss_exponent[:,:self.components]
        self.expectation_det_ln_lambda  = self.expectation_det_ln_lambda   [:self.components]
        self.alpha                      = self.alpha                       [:self.components]
        self.expectation_ln_pi          = self.expectation_ln_pi           [:self.components]
        self.r                          = self.r                         [:,:self.components]

        self.N_comp                     = self.N_comp                      [:self.components]
        self.nu                         = self.nu                          [:self.components]
        self.x_mean_comp                = self.x_mean_comp                 [:self.components]
        self.S                          = self.S                           [:self.components]
        self.beta                       = self.beta                        [:self.components]
        self.m                          = self.m                           [:self.components]
        self.W                          = self.W                           [:self.components]

    def likelihood_bound(self):
        # todo easy to parallize sum of independent terms
        bound = 0
        bound += self._expect_p_log_X() # (10.71)
        bound += self._expect_p_Z() # (10.72)
        C = Dirichlet_C(self.alpha)
        bound += self._expect_log_P_pi(C) # (10.73)
        bound += self._expect_log_P_mu_lambda() # (10.74)
        bound += self._expect_log_q_Z() # (10.75)
        bound += self._expect_log_q_pi(C) # (10.76)
        bound += self._expect_log_q_mu_lambda() # (10.77)
        return bound

    def _expect_log_P_X(self):
        # (10.71)

        result = 0
        for k in range(self.components):
            tmp = self.x_mean_comp[k] - self.m[k]
            result += self.N_comp[k] * \
                      (self.expectation_det_ln_lambda[k] - self.dim / self.beta[k]
                       -self.nu[k] * (_np.trace(self.S[k].dot(self.W[k]))
                       + tmp.dot(self.W[k]).dot(tmp)) - self.dim * log(2 * _np.pi))

        return 0.5 * result

    def _expect_P_Z(self):
        # (10.72)

        # contract all indices, no broadcasting
        return _np.einsum('nk,k->', self.r, self.expectation_ln_pi)

    def _expect_log_P_pi(self, log_C=None):
        # (10.73)

        if log_C is None:
            log_C = Dirichlet_log_C(self.alpha)

        return log_C + (self.alpha0 - 1) * _np.einsum('k->', self.expectation_ln_pi)

    def _expect_log_P_mu_lambda(self):
        # (10.74)

        result = 0
        for k in self.components:
            tmp = self.m[k] - self.m0
            result +=  self.expectation_det_ln_lambda[k] - self.dim * self.beta0 / self.beta[k] \
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

    def _expect_log_q_pi(self, log_C=None):
        # (10.76)

        if log_C is None:
            log_C = Dirichlet_log_C(self.alpha)

        return _np.einsum('k,k', self.alpha - 1, self.expectation_ln_pi) + log_C

    def _expect_log_q_mu_lambda(self):
        result = 0
        for k in self.components:
            result += 0.5 * (self.expectation_det_ln_lambda + self.dim * (log(self.beta[k] / (2 * _np.pi) - 1)))
            # Wishart entropy
            result -= Wishart_H(self.lam, W, nu, B)


class VBMerge(GaussianInference):
    # todo refer to set_variational_parameters for more kwargs
    """Parsimonious reduction of Gaussian mixture models with a
    variational-Bayes approach [BGP10]_

    The idea is to reduce the number of components of an overly complex
    Gaussian mixture while retaining an accurate description. The original
    samples are not required, hence it much faster compared to standard
    variational Bayes. The great advantage compared to hierarchical
    clustering is that the number of output components is chosen
    automatically. One starts with too many components and lets the
    algorithm remove unnecessary components.

    :param input_mixture:

        A Gaussian mixture density, the input to be compressed.

    :param initial_guess:

        A Gaussian mixture density, the starting point for the optimization.
        Its number of components defines the maximum possible.
        todo refer to ``GaussianMixture``

    :param N:

        The number of (virtual) input samples that the ``input_mixture`` is
        based on.

    """

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
        self.Nin = self.input.w * N

        self.set_variational_parameters(**kwargs)
        # take mean and covariances from initial guess
        if initial_guess is not None:
            # precision matrix is inverse of covariance
            for c in initial_guess:
                c._inv()
            self.W = _np.array([c.inv for c in initial_guess])

            # copy over the means
            self.m = _np.array([c.mean for c in initial_guess])

    def _initialize_output(self):
        GaussianInference._initialize_output(self)
        self.expectation_gauss_exponent = _np.zeros((self.L, self.components))

    def update(self, N=1):
        for i in range(N):
            # E-like step
            self._update_expectation_det_ln_lambda() # (21)
            self._update_expectation_ln_pi() # (22)
            self._update_expectation_gauss_exponent() # after (40)
            self._update_r() # after (40)

            # synthetic statistics
            _np.einsum('l,lk', self.Nin, self.r, out=self.N_comp) # (41)
            self._update_x_mean_comp() # (42)
            self._update_S() # (43) and (44)

            # M-like step
            self.alpha = self.alpha0 + self.N_comp # (45)
            self.beta = self.beta0 + self.N_comp # (46)
            self._update_m() # (47)
            self._update_W() # (48)
            self.nu = self.nu0 + self.N_comp # (49)

    def _update_expectation_gauss_exponent(self):
        for k in range(self.components):
            for l in range(len(self.input.comp)):
                tmp = self.input[l].mean - self.m[k]
                chi_squared = tmp.dot(self.W[k]).dot(tmp)
                self.expectation_gauss_exponent[l,k] = self.dim / self.beta[k] + self.nu[k] * \
                                                       (_np.trace(self.W[k].dot(self.input[l].inv)) + chi_squared)

    def _update_log_rho(self):
        # eqn 40 in [BGP10]
        # first line: compute k vector
        tmp_k  = 2 * self.expectation_ln_pi
        tmp_k += self.expectation_det_ln_lambda
        tmp_k -= self.dim * _np.log(2 * _np.pi)

        # turn into lk matrix
        self.log_rho = _np.einsum('l,k->lk', self.Nin, tmp_k)

        # add second line
        self.log_rho -= _np.einsum('l,lk->lk', self.Nin, self.expectation_gauss_exponent)

        self.log_rho /= 2.0

    def _update_S(self):
        # combine (43) and (44), since only ever need sum of S and C

        for k in range(self.components):
            self.S[k,:] = 0.0
            if self.N_comp[k] > 0:
                for l in range(self.L):
                    tmp        = self.mu[l] - self.x_mean_comp[k]
                    self.S[k] += self.Nin[l] * self.r[l,k] * (_np.outer(tmp, tmp) + self.input[l].cov)
                self.S[k] /= self.N_comp[k]

    def _update_x_mean_comp(self):
        _np.einsum('l,lk,li->ki', self.Nin, self.r, self.mu, out=self.x_mean_comp) # (42)
        # handle dead components, apply 1/N_k
        for k, nk in enumerate(self.N_comp):
            if nk > 0:
                self.x_mean_comp[k] /= nk

    def get_result(self):
        '''Return the variational mixture-density estimate'''

        # find mode of Wishart distribution (of precision)
        # and invert to find covariance
        # todo this is *not* the mode of the joint q*(\mu_k, \Lambda_k),
        # but only of W(\Lambda_k | ...), the Wishart distribution.
        # Hopefully it is fairly close
        #
        # The most likely value of the means is m, the mean parameter
        # of the Gaussian q(\mu_k)
        components = []
        weights = []
        for k, W in enumerate(self.W):
            if self.nu[k] >= self.dim + 1:
                cov = _np.linalg.inv((self.nu[k] - self.dim - 1) * W)
                components.append(GaussianMixture.Component(self.m[k], cov))
                # copy over precision matrix
                components[-1].inv = (self.nu[k] - self.dim - 1) * W
                weights.append(self.N_comp[k] / self.N)
            else:
                print('Skipping comp. %d with dof %g' % (k,self.nu[k]))

        return GaussianMixture(components, weights)

    def likelihood_bound(self):
        # todo use eqn labels from [BGP10]_ instead of 2001 paper
        pass

# todo move Wishart stuff to separate class, file?
def Wishart_log_B(W, nu, det=None):
    '''Compute first part of a Wishart distribution's normalization,
    (B.79) of [BGP10]_, on the log scale.

    :param W:

        Covariance matrix of a Wishart distribution.

    :param nu:

        Degrees of freedom of a Wishart distribution.

    :param det:

        The determinant of ``W``, :math:`|W|`. If `None`, recompute it.

    '''

    if det is None:
        det = _np.linalg.det(W)

#     B =  det**(-0.5 * nu)
#     B *= 2**(-0.5 * nu * len(W))
#     B *= _np.pi**(-0.25 * len(W) * (len(W) - 1))
#     for i in range(1, len(W) + 1):
#         B /= _gamma(0.5 * (nu + 1 - i))

    log_B = log(det**(-0.5 * nu))
    log_B -= 0.5 * nu * len(W) * log(2)
    log_B -= 0.25 * len(W) * (len(W) - 1) * log(_np.pi)
    for i in range(1, len(W) + 1):
        log_B -= _gammaln(0.5 * (nu + 1 - i))

    return log_B, det

def Wishart_H(Lambda, W, nu, log_B=None):
    '''Entropy of the Wishart distribution, (B.82) of [BGP10]_ .'''

    if log_B is None:
        log_B = Wishart_log_B(W, nu, det)

    conti

def Dirichlet_log_C(alpha):
    '''Compute normalization constant of Dirichlet distribution on
    log scale, (B.23) of [BGP10]_ .

    '''

    # compute gamma functions on log scale to avoid overflows
    log_C = _gammaln(_np.einsum('k->', alpha))
    for alpha_k in alpha:
        log_C -= _gammaln(alpha_k)

    return log_C
