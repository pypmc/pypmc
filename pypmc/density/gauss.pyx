"""Collect Gaussian probability densities"""

import numpy as _np
from .base import ProbabilityDensity, LocalDensity
from ..tools._doc import _inherit_docstring, _add_to_docstring

from pypmc.tools._linalg cimport bilinear_sym
from libc.math cimport log
cimport numpy as _np

class LocalGauss(LocalDensity):
    """A multivariate local Gaussian density with redefinable covariance.

    :param sigma:

         Matrix-like array; covariance-matrix.

    """
    symmetric = True
    def __init__(self, sigma):
        self.update(sigma)

    def update(self, sigma):
        """Re-initilize the proposal with a new covariance matrix.

        :param sigma:

            Matrix-like array; the new covariance-matrix.

        """
        self.sigma = _np.array(sigma)
        self.dim   = self.sigma.shape[0]
        self._sigma_decompose()

    def _sigma_decompose(self):
        """Private function to calculate the Cholesky decomposition, the
        inverse and the normalisation of the covariance matrix sigma and
        store it in the object instance

        """
        # first check if matrix is symmetric
        if not _np.allclose(self.sigma, self.sigma.transpose()):
            raise _np.linalg.LinAlgError('Matrix is not symmetric')
        self.cholesky_sigma = _np.linalg.cholesky(self.sigma)
        self.inv_sigma      = _np.linalg.inv(self.sigma)
        self.det_sigma      = _np.linalg.det(self.sigma)
        self._compute_norm()


    def _get_gauss_sample(self, rng):
        """transform sample from standard gauss to Gauss(mean=0, sigma=sigma)"""
        return _np.dot(self.cholesky_sigma, rng.normal(0,1,self.dim))

    def _compute_norm(self):
        'Compute the normalization'
        self.log_normalization = -.5 * self.dim * log(2 * _np.pi) - .5 * log(self.det_sigma)

    @_inherit_docstring(LocalDensity)
    def evaluate(self, _np.ndarray[double, ndim=1] x, _np.ndarray[double, ndim=1] y):
        return self.log_normalization - .5 * bilinear_sym(self.inv_sigma, x - y)

    @_add_to_docstring('''    .. important::\n
                ``rng`` must return a numpy array of N samples from:\n
                - **rng.normal(0,1,N)**: standard gaussian distribution\n''')
    @_inherit_docstring(LocalDensity)
    def propose(self, y, rng=_np.random.mtrand):
        return y + self._get_gauss_sample(rng)

class Gauss(ProbabilityDensity):
    r"""A Gaussian probability density. Can be used as component for
    MixtureDensities.

        :param mu:

            Vector-like array; the gaussian's mean :math:`\mu`

        :param sigma:

            Matrix-like array; the gaussian's covariance matrix :math:`\Sigma`

    """
    def __init__(self, mu, sigma):
        self.update(mu, sigma)

    def update(self, mu, sigma):
        r"""Re-initialize the density with new mean and covariance matrix.

        :param mu:

            Vector-like array; the gaussian's mean :math:`\mu`

        :param sigma:

            Matrix-like array; the gaussian's covariance matrix
            :math:`\Sigma`

        """
        self.mu          = _np.array(mu)
        self.dim         = len(self.mu)

        self._local_gauss = LocalGauss(sigma)

        self.inv_sigma   = self._local_gauss.inv_sigma # creates reference
        self.det_sigma   = self._local_gauss.det_sigma # creates copy because det_sigma is a float
        self.sigma       = self._local_gauss.sigma     # creates reference

        assert self.dim == self.sigma.shape[0], "Dimensions of mean (%d) and covariance matrix (%d) do not match!" %(self.dim,self.sigma.shape[0])

    @_inherit_docstring(ProbabilityDensity)
    def evaluate(self, _np.ndarray[double, ndim=1] x):
        return self._local_gauss.log_normalization - .5 * bilinear_sym(self.inv_sigma, x - self.mu)

    @_inherit_docstring(ProbabilityDensity)
    def multi_evaluate(self, _np.ndarray[double, ndim=2] x not None, _np.ndarray[double, ndim=1] out=None):
        if out is None:
            out = _np.empty(len(x))
        else:
            assert len(out) == len(x)

        cdef:
            double log_normalization = self._local_gauss.log_normalization
            double [:]   mu          = self.mu
            double [:]   diff        = _np.empty_like(x[0])
            double [:]   results     = out
            double [:,:] inv_sigma   = self.inv_sigma
            size_t       i, n

        for n in range(len(x)):
            # compute difference
            for i in range(len(diff)):
                diff[i] = x[n,i] - mu[i]

            results[n] = log_normalization - 0.5 * bilinear_sym(inv_sigma, diff)

        return out

    @_add_to_docstring("""    .. important::\n
                ``rng`` must meet the requirements of
                :py:meth:`.LocalGauss.propose`.\n\n""")
    @_inherit_docstring(ProbabilityDensity)
    def propose(self, int N=1, rng=_np.random.mtrand):
        output = _np.empty((N,self.dim))
        for i in range(N):
            output[i] = self._local_gauss.propose(self.mu)
        return output
