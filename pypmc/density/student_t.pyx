"""Collect Student's t probability densities"""

import numpy as _np
from scipy.special import gammaln as _gammaln
from .base import ProbabilityDensity
from .gauss import LocalGauss
from ..tools._doc import _inherit_docstring, _add_to_docstring

from pypmc.tools._linalg cimport bilinear_sym
from libc.math cimport log, sqrt
cimport numpy as _np

class LocalStudentT(LocalGauss):
    """A multivariate local Student's t density with redefinable covariance.

    :param sigma:

         Matrix-like array; the covariance-matrix


    :param dof:

         Float; the degrees of freedom

    """
    def __init__(self, sigma, double dof):
        self.symmetric = True
        assert dof > 0., "Degree of freedom (``dof``) must be greater than zero (got %g)." % dof
        self.dof       = dof
        self.update(sigma)

    def _compute_norm(self):
        self.log_normalization = _gammaln(.5 * (self.dof + self.dim)) - _gammaln(.5 * self.dof) \
                                 -0.5 * self.dim * log(self.dof * _np.pi) - 0.5 * self.log_det_sigma

    @_inherit_docstring(ProbabilityDensity)
    def evaluate(self, x , y):
        return self.log_normalization  - .5 * (self.dof + self.dim) \
            * log(1. + bilinear_sym(self.inv_sigma, x - y) / self.dof)

    @_add_to_docstring('''    .. important::\n
                ``rng`` must return\n
                - a numpy array of N samples from
                  **rng.normal(0,1,N)**: standard gaussian distribution
                - sample as float from
                  **rng.chisquare(degree_of_freedom)**: any chi-squared
                  distribution\n\n''')
    @_inherit_docstring(ProbabilityDensity)
    def propose(self, y, rng = _np.random.mtrand):
        # when Z is normally distributed with expected value 0 and std deviation sigma
        # and  V is chi-squared distributed with dof degrees of freedom
        # and  Z and V are independent
        # then Z*sqrt(dof/V) is t-distributed with dof degrees of freedom and std deviation sigma

        return y + self._get_gauss_sample(rng) * sqrt(self.dof / rng.chisquare(self.dof))

class StudentT(ProbabilityDensity):
    r"""A Student's t probability density. Can be used as a component in
    MixtureDensities.

    :param mu:

        Vector-like array; the gaussian's mean :math:`\mu`

    :param sigma:

        Matrix-like array; the gaussian's covariance matrix :math:`\Sigma`

    :param dof:

        Float; the degrees of freedom :math:`\nu`

    """
    def __init__(self, mu, sigma, double dof):
        self.update(mu, sigma, dof)
        self._tmp = _np.empty_like(self.mu)

    def update(self, mu, sigma, double dof):
        r"""
        Re-initialize the density with new mean, covariance matrix and
        degrees of freedom.

        :param mu:

            Vector-like array; the gaussian's mean :math:`\mu`

        :param sigma:

            Matrix-like array; the gaussian's covariance matrix
            :math:`\Sigma`

        :param dof:

            Float; the degrees of freedom :math:`\nu`

        .. note::

            On ``LinAlgError``, the old ``mu``, ``sigma``, and ``dof``
            are plugged in and the proposal remains in a valid state.

        """
        # first check if ``sigma`` is a valid covariance matrix
        new_local_t = LocalStudentT(sigma, dof)
        self._local_t = new_local_t

        self.mu        = _np.array(mu)
        self.dim       = len(self.mu)
        self.dof       = dof

        self.inv_sigma = self._local_t.inv_sigma
        self.log_det_sigma = self._local_t.log_det_sigma
        self.sigma     = self._local_t.sigma

        assert self.dim == self.sigma.shape[0], "Dimensions of mean (%d) and covariance matrix (%d) do not match!" %(self.dim,self.sigma.shape[0])

        self._eval_prefactor = - .5 * (self.dof + self.dim)
        self._inv_dof = 1. / self.dof

    @_inherit_docstring(ProbabilityDensity)
    def evaluate(self, _np.ndarray[double, ndim=1] x):
        cdef:
            size_t i, dim = self.dim
            double log_norm = self._local_t.log_normalization
            double inv_dof = self._inv_dof
            double prefactor = self._eval_prefactor
            double [:] tmp = self._tmp, mu = self.mu
            double [:,:] inv_sigma = self.inv_sigma

        for i in range(dim):
            tmp[i] = x[i] - mu[i]

        return log_norm  + prefactor * log(1. + bilinear_sym(self.inv_sigma, tmp) * inv_dof)

    @_inherit_docstring(ProbabilityDensity)
    def multi_evaluate(self, _np.ndarray[double, ndim=2] x not None, _np.ndarray[double, ndim=1] out=None):
        cdef:
            size_t       i, n
            size_t       N          = len(x)
            size_t       dim        = self.dim
            double       prefactor  = self._eval_prefactor
            double       inv_dof    = self._inv_dof
            double       log_norm   = self._local_t.log_normalization
            double [:]   mu         = self.mu
            double [:]   diff       = _np.empty_like(mu)
            double [:,:] inv_sigma  = self.inv_sigma

        if out is None:
            out = _np.empty(N)
        else:
            assert len(out) == N

        cdef double [:] results = out

        for n in range(N):
            # compute difference
            for i in range(len(diff)):
                diff[i] = x[n,i] - mu[i]

            results[n]  = bilinear_sym(self.inv_sigma, diff)
            results[n] *= inv_dof
            results[n] += 1.
            results[n]  = log(results[n])
            results[n] *= prefactor
            results[n] += log_norm

        return out

    @_add_to_docstring("""    .. important::\n
                ``rng`` must meet the requirements of
                :py:meth:`.LocalStudentT.propose`.\n\n""")
    @_inherit_docstring(ProbabilityDensity)
    def propose(self, int N=1, rng=_np.random.mtrand):
        output = _np.empty((N,self.dim))
        for i in range(N):
            output[i] = self._local_t.propose(self.mu, rng)
        return output
