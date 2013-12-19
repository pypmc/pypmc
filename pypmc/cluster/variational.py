"""Variational clustering as described in [Bis06]_

"""

from __future__ import division as _div
import numpy as _np
from scipy.special.basic import digamma as _digamma
from .._tools._doc import _inherit_docstring, _add_to_docstring

class _Inference(object):
    '''Abstract base class; approximates a probability density by a
    member of a specific class of probability densities

    '''

    def __init__(self):
        raise NotImplementedError('Do not create instances from this class, use derived classes instead.')

    def update(self, N = 1):
        '''Recalculates the parameters using the update equations

        :param N:

            An int which defines the maximum number of steps to run the
            iteration.

        '''
        raise NotImplementedError()

    def set_variational_parameters(self, *args, **kwargs):
        '''Resets the parameters to the submitted values or default
        Use this function to set initial values for the iteration.

        '''
        raise NotImplementedError()

    def prune(self, threshold = 1.):
        '''Deletes components with small effective number of samples :math:`N_k`

        :param threshold:

            Float; the minimum effective number of samples :math:`N_k`
            a component must have to survive.

        '''
        raise NotImplementedError()

    def get_result(self):
        '''Returns the parameters calculated by ``self.update``'''
        raise NotImplementedError()

class GaussianInference(_Inference):
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
        self.data       = _np.array(data) #call array constructor to be sure to have a copy
        self.components = components
        self.N,self.dim = self.data.shape

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

            Float; :math:`\beta_0` paramter for the probability distribution of
            :math:`\boldsymbol{\mu}` and :math:`\boldsymbol{\Lambda}`.

        :param nu0:

            Float; :math:`\nu_0` paramter for the probability distribution of
            :math:`\boldsymbol{\mu}` and :math:`\boldsymbol{\Lambda}`.

        :param m0:

            Matrix like array; :math:`m_0` paramter for the probability
            distribution of
            :math:`\boldsymbol{\mu}` and :math:`\boldsymbol{\Lambda}`.
            Provide an initial guess for the means here such that the
            mean of component i can be accessed as m0[i].

        :param W0:

            Matrix like array; :math:`\boldsymbol{W_0}` paramter for the
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

        self.alpha0 = kwargs.pop('alpha0', 1.e-10)

        # in the limit beta --> 0: uniform prior
        self.beta0  = kwargs.pop('beta0' , 1.e-20)

        # smallest possible nu such that the Wishart pdf does not diverge at 0
        self.nu0    = kwargs.pop('nu0'   , self.dim + 1.)


        self.m0     = kwargs.pop('m0'    , _np.zeros(self.dim * self.components).\
                                            reshape((self.components,self.dim)) )

        # covariance matrix; unit matrix <--> unknown correlation
        self.W0     = kwargs.pop('W0'    , None)
        if self.W0 == None:
            self.W0     = _np.eye(self.dim)
            self.inv_W0 = self.W0
        else:
            self.inv_W0 = _np.linalg.inv(self.W0)

        if not kwargs == {}: raise TypeError('unexpected keyword(s): ' + str(kwargs.keys()))

        self._initialize_output()

    def _initialize_output(self):
        '''Create all variables needed for the iteration in ``self.update``'''
        # starting x_mean_comp[k] all equal unables the algorithm to diverge into unequal components
        # thus do not use: self.x_mean_comp = self.m0.copy() if m0=np.zeros
        self.x_mean_comp = self.m0.copy() + _np.linspace(-1.,1., self.components*self.dim).reshape((self.components,self.dim))
        self.nu = _np.zeros(self.components) + (self.nu0 + 1)
        self.W = _np.array([self.W0 for i in range(self.components)])
        self.expectation_gauss_exponent = _np.zeros((  self.N,self.components  ))
        self.N_comp = _np.array([self.N/self.components for i in range(self.components)])
        self.beta = self.N_comp.copy() + self.beta0
        self.S = _np.empty_like(self.W)
        self.m = _np.empty((self.components,self.dim))

    # ------------------- below belongs to update ---------------------
    @_inherit_docstring(_Inference)
    def update(self, N=1):
        for i in range(N):
            #E-like-step
            self._update_expectation_gauss_exponent() #eqn 10.64 in [Bis06]
            self._update_expectation_det_ln_lambda() #eqn 10.65 in [Bis06]
            self.alpha = self.alpha0 + self.N_comp #eqn 10.58 in [Bis06]
            self.expectation_ln_pi = _digamma(self.alpha) - _digamma(self.alpha.sum()) #eqn 10.66 in [Bis06]
            self._update_r() #eqn 10.46 and 10.49 in [Bis06]

            #M-like-step
            self.N_comp = self.r.sum(axis = 0) #eqn 10.51 in [Bis06]
            self.nu = self.nu0 + self.N_comp + 1. #eqn 10.63 in [Bis06]
            self._update_x_mean_comp() #eqn 10.52 in [Bis06]
            self._update_S() #eqn 10.53 in [Bis06]
            self.beta = self.beta0 + self.N_comp #eqn 10.60 in [Bis06]
            self._update_m() #eqn 10.61 in [Bis06]
            self._update_W() #eqn 10.62 in [Bis06]

            # TODO: insert convergence criterion --> log-likelihood
            # TODO: implement support for weights

    def _update_m(self):
        for k in range(self.components):
            self.m[k] = 1./self.beta[k] * (self.beta0*self.m0[k] + self.N_comp[k]*self.x_mean_comp[k])

    def _update_r(self):
        unnormalized_r  = _np.exp(self.expectation_ln_pi + .5 * self.expectation_det_ln_lambda
                          - .5*self.dim*_np.log(2.*_np.pi) - .5*self.expectation_gauss_exponent)
        normalization_r = unnormalized_r.sum(axis=1).reshape((self.N,1))
        self.r          = unnormalized_r/normalization_r

    def _update_expectation_det_ln_lambda(self):
        logdet_W = _np.array([_np.log(_np.linalg.det(self.W[k])) for k in range(self.components)])

        tmp = 0
        for i in range(1,self.dim+1):
            tmp += _digamma(    .5*(self.nu + 1. - i )    )

        self.expectation_det_ln_lambda = tmp + self.dim*_np.log(2.) + logdet_W

    def _update_expectation_gauss_exponent(self): #_expectation_gauss_exponent --> _expectation_gauss_exponent[n,k]
        for k in range(self.components):
            for n in range(self.N):
                tmp                                  = _np.array([self.data[n] - self.x_mean_comp[k]])
                self.expectation_gauss_exponent[n,k] = self.dim/self.beta[k] + self.nu[k]*_np.dot(tmp,_np.dot(self.W[k],tmp.T))

    def _update_x_mean_comp(self):
        for k in range(self.components):
            if not self.N_comp[k] == 0: # prevent errors and x_mean is unimportant for a dead component
                self.x_mean_comp[k] = 1./self.N_comp[k] * (self.r[:,k] * self.data.T).T.sum(axis = 0)

    def _update_S(self):
        self.S = _np.zeros_like(self.S)
        for k in range(self.components):
            for n in range(self.N):
                if not self.N_comp[k] == 0: # prevent errors and S for a dead component is unimportant
                    tmp        = _np.array([self.data[n] - self.x_mean_comp[k]])
                    self.S[k] += 1./self.N_comp[k] * self.r[n,k] * _np.dot(tmp.T,tmp)

    def _update_W(self):
        for k in range(self.components):
            tmp = _np.array([self.x_mean_comp[k] - self.m0[k]])
            new_Wk = self.inv_W0 + self.N_comp[k]*self.S[k] +\
                     (self.beta0*self.N_comp[k])/(self.beta0+self.N_comp[k]) * _np.dot(tmp.T,tmp)
            self.W[k] = _np.linalg.inv(new_Wk)

    # ------------------- above belongs to update ---------------------

    def get_result(self):
        '''Returns the parameters calculated by ``self.update`` as
        tuple(abundances, means, covariances)

        '''
        return self.N_comp/self.N,self.m,self.S

    @_inherit_docstring(_Inference)
    def prune(self, threshold = 1.):
        components_to_survive = _np.where(self.N_comp>=threshold)[0]
        self.components = len(components_to_survive)

        self.expectation_gauss_exponent = _np.empty((  self.N,self.components  ))

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
