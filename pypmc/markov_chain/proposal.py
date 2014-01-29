"""Collect MCMC proposal densities"""

import numpy as _np
from scipy.special import gammaln as _gammaln
from .._tools._doc import _inherit_docstring, _add_to_docstring

class ProposalDensity(object):
    """A proposal density for a local-random-walk Markov chain sampler.

    """

    symmetric = False

    def __init__(self):
        raise NotImplementedError('Do not create instances from this class, use derived classes instead.')

    def evaluate(self, x, y):
        """Evaluate log of the density to propose ``x`` given ``y``, namely log(q(x|y)).

        :param x:

            The proposed point.

        :param y:

            The current point of the Markov chain.

        """
        raise NotImplementedError()

    def propose(self, y, rng = _np.random.mtrand):
        """propose(self, y, rng = numpy.random.mtrand)
        Propose a new point given ``y`` using the random number
        generator ``rng``.

        :param y:

            The current position of the chain.

        :param rng:

            The state of a random number generator like numpy.random.mtrand

        """
        raise NotImplementedError()

class Multivariate(ProposalDensity):
    """Abstract multivariate proposal density with updatable covariance.
    Do not create instances from this class, use derived classes instead.

    """
    def __init__(self, sigma):
        raise NotImplementedError('Do not create instances from this class, use derived classes instead.')

    def update_sigma(self, sigma):
        """Re-initiate the proposal with new covariance matrix.

        :param sigma:

            A numpy array representing the covariance-matrix.

        """
        self.dim   = sigma.shape[0]
        self.sigma = sigma.copy()
        self._sigma_decompose()


    def _sigma_decompose(self):
        """Private function to calculate the Cholesky decomposition, the
        inverse and the normalisation of the covariance matrix sigma and
        store it in the object instance

        """
        self.cholesky_sigma = _np.linalg.cholesky(self.sigma)
        self.inv_sigma      = _np.linalg.inv(self.sigma)
        self._compute_norm()

    def _compute_norm(self):
        """Private function to calculate the normalisation of the
        covariance matrix sigma and store it in the object instance

        """
        raise NotImplementedError()

    def _get_gauss_sample(self, rng):
        """transform sample from standard gauss to Gauss(mean=0, sigma = sigma)"""
        return _np.dot(self.cholesky_sigma,rng.normal(0,1,self.dim))

class MultivariateGaussian(Multivariate):
    """A multivariate Gaussian density with redefinable covariance.

    :param sigma:

         A numpy array representing the covariance-matrix.

    """
    def __init__(self, sigma):
        self.symmetric = True
        super(MultivariateGaussian, self).update_sigma(sigma)

    def _compute_norm(self):
        self.log_normalization = -.5 * self.dim * _np.log(2 * _np.pi) + .5 * _np.log(_np.linalg.det(self.inv_sigma))

    @_inherit_docstring(ProposalDensity)
    def evaluate(self, x , y):
        return self.log_normalization - .5 * _np.dot(_np.dot(x-y, self.inv_sigma), x-y)

    @_add_to_docstring('''    .. important::\n
                ``rng`` must return a numpy array of N samples from:\n
                - **rng.normal(0,1,N)**: standard gaussian distribution\n''')
    @_inherit_docstring(ProposalDensity)
    def propose(self, y, rng = _np.random.mtrand):
        return y + self._get_gauss_sample(rng)

class MultivariateStudentT(Multivariate):
    """A multivariate Student-t density with redefinable covariance.

    :param sigma:

         A numpy array representing the covariance-matrix.


    :param dof:

         A float or int representing the degree of freedom.

    """

    def __init__(self, sigma, dof):
        self.symmetric = True
        self.dof       = dof
        super(MultivariateStudentT, self).update_sigma(sigma)

    def _compute_norm(self):
        self.log_normalization = _gammaln(.5 * (self.dof + self.dim)) - _gammaln(.5 * self.dof) \
                                 -0.5 * self.dim * _np.log(self.dof * _np.pi) + .5 * _np.log(_np.linalg.det(self.inv_sigma))

    @_inherit_docstring(ProposalDensity)
    def evaluate(self, x , y):
        return self.log_normalization  - .5 * (self.dof + self.dim) \
            * _np.log(1. + (_np.dot(_np.dot(x-y, self.inv_sigma), x-y)) / self.dof)

    @_add_to_docstring('''    .. important::\n
                ``rng`` must return a numpy array of N samples from:\n
                - **rng.normal(0,1,N)**: standard gaussian distribution
                - **rng.chisquare(degree_of_freedom, N)**: any chi-squared distribution''')
    @_inherit_docstring(ProposalDensity)
    def propose(self, y, rng = _np.random.mtrand):
        # when Z is normally distributed with expected value 0 and std deviation sigma
        # and  V is chi-squared distributed with dof degrees of freedom
        # and  Z and V are independent
        # then Z*sqrt(dof/V) is t-distributed with dof degrees of freedom and std deviation sigma

        return y + self._get_gauss_sample(rng) * _np.sqrt(self.dof / rng.chisquare(self.dof))
