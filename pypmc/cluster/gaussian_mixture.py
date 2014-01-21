import numpy as np
import scipy.linalg as linalg

# todo document members
class GaussianMixture(object):
    """A Gaussian mixture density.

    :param components:

        An iterable of ``Component``.

    :param weights:

        The weight associated with each component. If ``None``, assign
        equal weight to every component. If given, the weights are
        normalized to sum to one.

    """
    def __init__(self, components, weights=None):
        self.comp = list(components)
        if weights is None:
            self.w = np.ones(len(self.comp))
        else:
            assert len(weights) == len(self.comp)
            self.w = np.array(weights)
        self.normalize()

    def __getitem__(self, i):
        return self.comp[i]

    def normalize(self):
        """Normalize the component weights to sum up to 1."""
        self.w /= self.w.sum()

    def normalized(self):
        try:
            # precision loss in sum of many small numbers, so can't
            # expect 1.0 exactly
            np.testing.assert_allclose(self.w.sum(), 1.0)
            return True
        except AssertionError:
            return False

    def prune(self):
        """Remove components with vanishing weight.
        Return list of indices of removed components.

        """
        # go reverse s.t. indices remain valid
        removed_indices = []
        n = len(self.comp)
        for i, c in enumerate(reversed(self.comp)):
            if not self.w[n - i - 1]:
                removed_indices.append(n - i - 1)
                self.comp.pop(removed_indices[-1])

        # adjust weights
        if removed_indices:
            self.w = np.delete(self.w, removed_indices)

        return removed_indices

    class Component(object):
        """Minimal description of a Gaussian component in a mixture density.

        :param weight:

            the component weight in [0,1]

        :param mean:

            vector-like

        :param cov:

            matrix-like

        :param inv:

            Compute inverse and determinant of ``cov``.

        :attribute det:

            Determinant of the covariance.

        :attribute inv:

            Inverse of the covariance. Only available if ``inv``
            enabled at start up.

        """
        def __init__(self, mean, cov, inv=False):
            self.mean = mean
            self.cov = cov

            self._verify()

            self._det()
            if inv:
                self._inv()

        def _verify(self):
            assert len(self.mean) > 0
            # dimensions have to match
            assert(self.cov.shape == (len(self.mean), len(self.mean)))

        def _det(self):
            """Compute determinant of covariance"""
            # determinant of triangular matrix just product of diagonal elements
            self.det = linalg.det(self.cov)

        def _inv(self):
            """Compute inverse of covariance"""

            self.inv = linalg.inv(self.cov)

        def recompute_det_inv(self):
            """Compute inverse and determinant of ``cov``."""
            # todo could be better with Cholesky but it seems no easy computation of inverse from L
            # L = np.linalg.cholesky(self.cov)

            self._inv()
            self._det()

