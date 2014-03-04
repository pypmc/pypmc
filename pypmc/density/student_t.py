"""Collect StudentT probability densities"""

import numpy as _np
from scipy.special import gammaln as _gammaln
from .base import ProbabilityDensity
from .gauss import LocalGauss
from ..tools._doc import _inherit_docstring, _add_to_docstring

class LocalStudentT(LocalGauss):
    """A multivariate local StudentT density with redefinable covariance.

    :param sigma:

         Matrix-like array; the covariance-matrix


    :param dof:

         Float or Integer; the degrees of freedom

    """

    def __init__(self, sigma, dof):
        self.symmetric = True
        self.dof       = dof
        self.update(sigma)

    def _compute_norm(self):
        self.log_normalization = _gammaln(.5 * (self.dof + self.dim)) - _gammaln(.5 * self.dof) \
                                 -0.5 * self.dim * _np.log(self.dof * _np.pi) + .5 * _np.log(_np.linalg.det(self.inv_sigma))

    @_inherit_docstring(ProbabilityDensity)
    def evaluate(self, x , y):
        return self.log_normalization  - .5 * (self.dof + self.dim) \
            * _np.log(1. + (_np.dot(_np.dot(x-y, self.inv_sigma), x-y)) / self.dof)

    @_add_to_docstring('''    .. important::\n
                ``rng`` must return a numpy array of N samples from:\n
                - **rng.normal(0,1,N)**: standard gaussian distribution
                - **rng.chisquare(degree_of_freedom, N)**: any chi-squared
                  distribution''')
    @_inherit_docstring(ProbabilityDensity)
    def propose(self, y, rng = _np.random.mtrand):
        # when Z is normally distributed with expected value 0 and std deviation sigma
        # and  V is chi-squared distributed with dof degrees of freedom
        # and  Z and V are independent
        # then Z*sqrt(dof/V) is t-distributed with dof degrees of freedom and std deviation sigma

        return y + self._get_gauss_sample(rng) * _np.sqrt(self.dof / rng.chisquare(self.dof))

class StudentT(ProbabilityDensity):
    r"""A Student T probability density. Can be used as component for
    MixtureDensities.

        :param mu:

            Vector-like array; the gaussian's mean :math:`\mu`

        :param sigma:

            Matrix-like array; the gaussian's covariance matrix :math:`\Sigma`

        :param dof:

            Float; the degrees of freedom :math:`\nu`

    """
    def __init__(self, mu, sigma, dof):
        self.update(mu, sigma, dof)

    def update(self, mu, sigma, dof):
        r"""Re-initialize the density with new mean, covariance matrix and
        degrees of freedom.

        :param mu:

            Vector-like array; the gaussian's mean :math:`\mu`

        :param sigma:

            Matrix-like array; the gaussian's covariance matrix
            :math:`\Sigma`

        :param dof:

            Float; the degrees of freedom :math:`\nu`

        """
        self.mu        = _np.array(mu)
        self.dim       = len(self.mu)
        self.dof       = dof

        self._local_t  = LocalStudentT(sigma, dof)

        self.inv_sigma = self._local_t.inv_sigma
        self.det_sigma = self._local_t.det_sigma
        self.sigma     = self._local_t.sigma

        assert self.dim == self.sigma.shape[0], "Dimensions of mean (%d) and covariance matrix (%d) do not match!" %(self.dim,self.sigma.shape[0])

    @_inherit_docstring(ProbabilityDensity)
    def evaluate(self, x):
        return self._local_t.evaluate(x,self.mu)

    @_add_to_docstring("""    .. important::\n
                ``rng`` must meet the requirements of
                :py:meth:`.LocalStudentT.propose`.\n\n""")
    @_inherit_docstring(ProbabilityDensity)
    def propose(self, N=1, rng=_np.random.mtrand):
        output = _np.empty((N,self.dim))
        for i in range(N):
            output[i] = self._local_t.propose(self.mu)
        return output
