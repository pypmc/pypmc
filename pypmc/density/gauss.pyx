"""Collect Gaussian probability densities"""

import numpy as _np
from .base import ProbabilityDensity, LocalDensity
from ..tools._doc import _inherit_docstring, _add_to_docstring

from pypmc.tools._linalg cimport bilinear_sym, chol_inv_det
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
        r"""
        Re-initialize the proposal with a new covariance matrix.

        :param sigma:

            Matrix-like array; the new covariance-matrix.

        .. note::

            On ``LinAlgError``, the old covariance matrix is plugged in
            and the proposal remains in a valid state.

        """
        # turn scalar into 1x1 matrix if needed
        sigma = _np.asarray(_np.matrix(sigma, dtype=float, copy=True))

        # If ``sigma`` is invalid, the following line raises ``LinAlgError``
        # and the proposal remains in the old state. In particular, no internal
        # variable is changed.
        self.cholesky_sigma, self.inv_sigma, self.log_det_sigma = chol_inv_det(sigma)

        # update internals only if consistency checks inside ``chol_inv_det`` pass
        self.sigma = sigma
        self.dim = self.sigma.shape[0]
        self._compute_norm()

    def _get_gauss_sample(self, rng):
        """transform sample from standard gauss to Gauss(mean=0, sigma=sigma)"""
        return _np.dot(self.cholesky_sigma, rng.normal(0, 1, self.dim))

    def _compute_norm(self):
        'Compute the normalization'
        self.log_normalization = -0.5 * self.dim * log(2 * _np.pi) - 0.5 * self.log_det_sigma

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
        self._tmp = _np.empty_like(self.mu)

    def update(self, mu, sigma):
        r"""
        Re-initialize the density with new mean and covariance matrix.

        :param mu:

            Vector-like array; the gaussian's mean :math:`\mu`

        :param sigma:

            Matrix-like array; the gaussian's covariance matrix
            :math:`\Sigma`

        .. note::

            On ``LinAlgError``, the old ``mu`` and ``sigma`` are plugged
            in and the proposal remains in a valid state.

        """
        # first check if ``sigma`` is a valid covariance matrix
        new_local_gauss = LocalGauss(sigma)
        self._local_gauss = new_local_gauss

        self.mu          = _np.array(mu)
        self.dim         = len(self.mu)

        self.inv_sigma     = self._local_gauss.inv_sigma # creates reference
        self.log_det_sigma = self._local_gauss.log_det_sigma # creates copy because det_sigma is a float
        self.sigma         = self._local_gauss.sigma     # creates reference

        assert self.dim == self.sigma.shape[0], "Dimensions of mean (%d) and covariance matrix (%d) do not match!" %(self.dim,self.sigma.shape[0])

    @_add_to_docstring('\n        ' + ProbabilityDensity.evaluate.__doc__)
    def evaluate(self, _np.ndarray[double, ndim=1] x):
        cdef:
            size_t i, dim = self.dim
            double log_norm = self._local_gauss.log_normalization
            double [:] tmp = self._tmp, mu = self.mu
            double [:,:] inv_sigma = self.inv_sigma

        for i in range(dim):
            tmp[i] = x[i] - mu[i]

        return log_norm - .5 * bilinear_sym(inv_sigma, tmp)

    @_add_to_docstring('\n        ' + ProbabilityDensity.multi_evaluate.__doc__)
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

    @_add_to_docstring('\n        ' + ProbabilityDensity.propose.__doc__ +
                       """.. important::\n
            ``rng`` must meet the requirements of
            :py:meth:`.LocalGauss.propose`.\n\n""")
    def propose(self, int N=1, rng=_np.random.mtrand):
        output = _np.empty((N,self.dim))
        for i in range(N):
            output[i] = self._local_gauss.propose(self.mu, rng)
        return output
