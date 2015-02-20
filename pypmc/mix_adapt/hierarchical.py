"""Hierarchical clustering as described in [GR04]_

"""
import copy
import numpy as np
import scipy.linalg as linalg

class Hierarchical(object):
    """Hierarchical clustering as described in [GR04]_.

    Find a Gaussian mixture density :math:`g` with components
    :math:`g_j` that most closely matches the Gaussian mixture density
    specified by :math:`f` and its components :math:`f_i`, but with
    less components. The algorithm is an iterative EM procedure
    alternating between a *regroup* and a *refit* step, and requires
    an ``initial_guess`` of the output density that defines the
    maximum number of components to use.

    :param input_components:

        :py:class:`pypmc.density.mixture.MixtureDensity` with Gaussian
        (:py:class:`pypmc.density.gauss.Gauss`) components; the Gaussian
        mixture to be reduced.

    :param initial_guess:

        :py:class:`pypmc.density.mixture.MixtureDensity` with Gaussian
        (:py:class:`pypmc.density.gauss.Gauss`) components; initial guess
        for the EM algorithm.

    .. seealso::

        :py:func:`pypmc.density.mixture.create_gaussian_mixture`

    """
    def __init__(self, input_components, initial_guess):

        # read and verify component numbers
        self.nin = len(input_components.components)
        self.nout = len(initial_guess.components)

        assert self.nin > self.nout, "Got more output (%i) than input (%i) components" % (self.nout, self.nin)
        assert self.nout > 0, "Invalid number of output components %s" % self.nout

        self.f = input_components
        self.g = copy.deepcopy(initial_guess)

        # inverse map: several inputs can map to one output, so need list
        self.inv_map = {}
        for j in range(self.nout):
            self.inv_map[j] = None

        # the i-th element is :math:`min_j KL(f_i || g_j)`
        self.min_kl = np.zeros(self.nin) + np.inf

    def _cleanup(self, kill, verbose):
        """Look for dead components (weight=0) and remove them
        if enabled by ``kill``.
        Resize storage. Recompute determinant and covariance.

        """
        if kill:
            removed_indices = self.g.prune()

            self.nout -= len(removed_indices)

            if verbose and removed_indices:
                print('Removing %s' % removed_indices)

            for j in removed_indices:
                self.inv_map.pop(j[0])

    def _distance(self):
        """Compute the distance function d(f,g,\pi), Eq. (3)"""
        return np.average(self.min_kl, weights=self.f.weights)

    def _refit(self):
        """Update the map :math:`\pi` keeping the output :math:`g` fixed

        Use Eq. (7) and below in [GR04]_

        """
        # temporary variables for manipulation
        mu_diff = np.empty_like(self.f.components[0].mu)
        sigma   = np.empty_like(self.f.components[0].sigma)
        mean    = np.empty_like(mu_diff)
        cov     = np.empty_like(sigma)

        for j, c in enumerate(self.g.components):
            # stop if inv_map is empty for j-th comp.
            if not self.inv_map[j]:
                self.g.weights[j] = 0.
                continue

            # (re-)initialize new mean/cov to zero
            mean[:] = 0.0
            cov[:] = 0.0

            # compute total weight and mean
            self.g.weights[j] = self.f.weights[self.inv_map[j]].sum()
            for i in self.inv_map[j]:
                mean += self.f.weights[i] * self.f.components[i].mu

            # rescale by total weight
            mean /= self.g.weights[j]

            # update covariance
            for i in self.inv_map[j]:
                # mu_diff = mu'_j - mu_i
                mu_diff[:] = mean
                mu_diff -= self.f.components[i].mu

                # sigma = (mu'_j - mu_i) (mu'_j - mu_i)^T
                sigma[:] = np.outer(mu_diff, mu_diff)

                # sigma += sigma_i
                sigma += self.f.components[i].sigma

                # multiply with alpha_i
                sigma *= self.f.weights[i]

                # sigma_j += alpha_i * (sigma_i + (mu'_j - mu_i) (mu'_j - mu_i)^T
                cov += sigma

            # 1 / beta_j
            cov /= self.g.weights[j]

            # update the Mixture
            c.update(mean, cov)

    def _regroup(self):
        """Update the output :math:`g` keeping the map :math:`\pi` fixed.
        Compute the KL between all input and output components.

        """
        # clean up old maps
        for j in range(self.nout):
            self.inv_map[j] = []

        # find smallest divergence between input component i
        # and output component j of the cluster mixture density
        for i in range(self.nin):
            self.min_kl[i] = np.inf
            j_min = None
            for j in range(self.nout):
                kl = kullback_leibler(self.f.components[i], self.g.components[j])
                if kl < self.min_kl[i]:
                    self.min_kl[i] = kl
                    j_min = j
            assert j_min is not None
            self.inv_map[j_min].append(i)

    def run(self, eps=1e-4, kill=True, max_steps=50, verbose=False):
        r"""Perform the clustering on the input components updating the initial
        guess. The result is available in the member ``self.g``.

        Return the number of iterations at convergence, or None.

        :param eps:

            If relative change of distance between current and last step falls below ``eps``,
            declare convergence:

            .. math::
                0 < \frac{d^t - d^{t-1}}{d^t} < \varepsilon

        :param kill:

             If a component is assigned zero weight (no input components), it is removed.

        :param max_steps:

             Perform a maximum number of update steps.

        :param verbose:

             Output information on progress of algorithm.

        """
        old_distance = np.finfo(np.float64).max
        new_distance = np.finfo(np.float64).max

        if verbose:
            print('Starting hierarchical clustering with %d components.' % len(self.g.components))
        converged = False
        for step in range(1, max_steps + 1):
            self._cleanup(kill, verbose)
            self._regroup()
            self._refit()

            new_distance = self._distance()
            assert new_distance >= 0, 'Found non-positive distance %d' % new_distance

            if verbose:
                print('Distance in step %d: %g' % (step, new_distance))
            if new_distance == old_distance:
                converged = True
                if verbose:
                    print('Exact minimum found after %d steps' % step)
                break

            rel_change = (old_distance - new_distance) / old_distance
            assert not (rel_change < -1e-13), 'distance increased'

            if rel_change < eps and not converged and step > 0:
                converged = True
                if verbose and new_distance != old_distance:
                    print('Close enough to local minimum after %d steps' % step)
                break

            # save distance for comparison in next step
            old_distance = new_distance

        self._cleanup(kill, verbose)
        if verbose:
            print('%d components remain.' % len(self.g.components))

        if converged:
            return step
        # else return None

def kullback_leibler(c1, c2):
    """Kullback Leibler divergence of two Gaussians, :math:`KL(1||2)`"""
    d = c2.log_det_sigma - c1.log_det_sigma
    d += np.trace(c2.inv_sigma.dot(c1.sigma))
    mean_diff = c1.mu - c2.mu
    d += mean_diff.transpose().dot(c2.inv_sigma).dot(mean_diff)
    d -= len(c1.mu)

    return 0.5 * d
