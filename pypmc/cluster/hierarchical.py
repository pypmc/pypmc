"""Hierarchical clustering as described in [GR04]_

"""
import copy
import numpy as np
import scipy.linalg as linalg

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

    def __iter(self):
        for c in self.comp:
            yield c

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

class Hierarchical(object):
    """Hierarchical clustering as described in [GR04]_.

    Find a Gaussian mixture density with components :math:`g_j` that
    most closely matches the Gaussian mixture density specified by
    ``f`` and its components :math:`f_i`, but with less
    components. The algorithm is an iterative EM procedure alternating
    between a *regroup* and a *refit* step, and requires an
    ``initial_guess`` of the output density that defines the maximum
    number of components to use.

    :param input_components:

        :py:class:`pypmc.cluster.hierarchical.GaussianMixture`, the Gaussian
        mixture to be reduced.

    :param initial_guess:

        :py:class:`pypmc.cluster.hierarchical.GaussianMixture`, initial guess for
        the EM algorithm.

    :param verbose:

        Output information on progress of algorithm.

    """
    def __init__(self, input_components, initial_guess, verbose=False):

        # read and verify component numbers
        self.nin = len(input_components.comp)
        self.nout = len(initial_guess.comp)

        assert self.nin > self.nout, "Cannot reduce number of input components: %s" % (self.nin, self.nout)
        assert self.nout > 0, "Invalid number of output components %s" % self.nout

        self.f = input_components
        self.g = copy.copy(initial_guess)

        self._verbose = verbose

        # inverse map: several inputs can map to one output, so need list
        self.inv_map = {}
        for j in range(self.nout):
            self.inv_map[j] = None

        # the i-th element is :math:`min_j KL(f_i || g_j)`
        self.min_kl = np.empty(self.nin) + np.inf

    def _cleanup(self, kill):
        """Look for dead components (weight=0) and remove them
        if enable by ``kill``.
        Resize storage. Recompute determinant and covariance.

        """
        removed_indices = []
        if kill:
            removed_indices = self.g.prune()

            self.nout -= len(removed_indices)

            if self._verbose and removed_indices:
                print('Removing %s' % removed_indices)

        for j in removed_indices:
            self.inv_map.pop(j)

        # covariance and determinant need to be (re-)calculated
        for c in self.g:
            c.recompute_det_inv()

    def _distance(self):
        """Compute the distance function d(f,g,\pi), Eq. (3)"""
        return np.average(self.min_kl, weights=self.f.w)

    def _refit(self):
        """Update the map :math:`\pi` keeping the output :math:`g` fixed

        Use Eq. (7) and below in [GR04]_

        """
        # todo parallelize

        # temporary variables for manipulation
        mu_diff = np.empty_like(self.f[0].mean)
        sigma = np.empty_like(self.f[0].cov)

        for j, c in enumerate(self.g):
            # stop if inv_map is empty for j-th comp.
            if not self.inv_map[j]:
                continue

            # update in place
            c.mean[:] = 0.0
            c.cov[:] = 0.0

            # compute total weight and mean
            weight = self.f.w[self.inv_map[j]].sum()
            for i in self.inv_map[j]:
                c.mean += self.f.w[i] * self.f[i].mean

            # rescale by total weight
            c.mean /= weight

            # update covariance
            for i in self.inv_map[j]:
                # mu_diff = mu'_j - mu_i
                mu_diff[:] = c.mean
                mu_diff -= self.f[i].mean

                # sigma = (mu'_j - mu_i) (mu'_j - mu_i)^T
                sigma[:] = np.outer(mu_diff, mu_diff)

                # sigma += sigma_i
                sigma += self.f[i].cov

                # multiply with alpha_i
                sigma *= self.f.w[i]

                # sigma_j += alpha_i * (sigma_i + (mu'_j - mu_i) (mu'_j - mu_i)^T
                c.cov += sigma
            # 1 / beta_j
            c.cov /= weight

    def _regroup(self):
        """Update the output :math:`g` keeping the map :math:`\pi` fixed.
        Compute the KL between all input and output components.

        """
        # clean up old maps
        for j in range(self.nout):
            self.inv_map[j] = []

        # todo parallelize
        # find smallest divergence between input component i
        # and output component j of the cluster mixture density
        for i in range(self.nin):
            self.min_kl[i] = np.inf
            j_min = None
            for j in range(self.nout):
                kl = kullback_leibler(self.f[i], self.g[j])
                if kl < self.min_kl[i]:
                    self.min_kl[i] = kl
                    j_min = j
            assert j_min is not None
            self.inv_map[j_min].append(i)

    def run(self, eps=1e-4, kill=True, max_steps=50):
        r"""Perform the clustering on the input components
        updating the initial guess.

        :param eps:
            If relative change of distance between current and last step falls below ``eps``,
            declare convergence:

            .. math::
                0 < \frac{d^t - d^{t-1}}{d^t} < \varepsilon

        :param kill:
             If a component is assigned zero weight (no input components), it is removed.

        :param max_steps:
             Perform a maximum number of update steps.

        """
        old_distance = np.finfo(np.float64).max
        new_distance = np.finfo(np.float64).max

        if self._verbose:
            print('Starting hierarchical clustering with %d components.' % len(self.g.comp))
        converged = False
        step = 0
        while not converged and step < max_steps:
            self._cleanup(kill)
            self._regroup()
            self._refit()

            new_distance = self._distance()
            assert new_distance >= 0, 'Found non-positive distance %d' % new_distance

            if self._verbose:
                print('Distance in step %d: %g' % (step, new_distance))
            if new_distance == old_distance:
                coverged = True
                if self._verbose:
                    print('Exact minimum found after %d steps' % step)

            rel_change = (old_distance - new_distance) / old_distance
            assert not (rel_change < -1e-13), 'distance increased'

            if rel_change < eps and not converged and step > 0:
                converged = True
                if self._verbose and new_distance != old_distance:
                    print('Close enough to local minimum after %d steps' % step)

            # save distance for comparison in next step
            old_distance = new_distance

            step += 1

        assert converged
        self._cleanup(kill)
        if self._verbose:
            print('%d components remain.' % len(self.g.comp))

        return self.g

def kullback_leibler(c1, c2):
    """Kullback Leibler divergence of two Gaussians, :math:`KL(1||2)`"""
    # todo improve speed with scipy.linalg.blas and preallocating vector, matrix of right dim
    d = np.log(c2.det / c1.det)
    d += np.trace(c2.inv.dot(c1.cov))
    mean_diff = c1.mean - c2.mean
    d += mean_diff.transpose().dot(c2.inv).dot(mean_diff)
    d -= len(c1.mean)

    return 0.5 * d
