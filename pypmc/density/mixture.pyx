"""Collect mixture probability densities"""

import numpy as _np
from copy import deepcopy as _deepcopy
from .base import ProbabilityDensity
from .gauss import Gauss
from .student_t import StudentT
from ..tools._doc import _inherit_docstring, _add_to_docstring

cimport numpy as _np
from pypmc.tools._regularize cimport logsumexp, logsumexp2D

_msg_expect_normalized_weights = \
    """.. important::

            This function assumes that the weights are normalized, i.e.
            that ``self.normalized()`` returns ``True``.

    """

class MixtureDensity(ProbabilityDensity):
    """Base class for mixture probability densities.

    :param components:

        Iterable of ProbabilityDensities; the mixture's components

    :param weights:

        Iterable of floats; the weights of the components
        (will be normalized automatically during initialization)

    """
    def __init__(self, components, weights=None):
        self.components  = [_deepcopy(component) for component in components]

        assert self.components, "Must have at least one component!"

        # assertion above prevents this from causing segmentation fault
        self.dim = self.components[0].dim

        _np.testing.assert_equal([comp.dim for comp in self.components],[self.dim for comp in components])

        if weights is None:
            self.weights = _np.ones(len(self.components))
        else:
            self.weights = _np.array(weights, dtype=float)
            assert len(self.weights) == len(self.components)

        self.normalize()

    def __len__(self):
        number_of_components = len(self.components)
        assert number_of_components == len(self.weights)
        return number_of_components

    def normalize(self):
        """Normalize the component weights to sum up to 1."""
        self.weights /= self.weights.sum()

    def normalized(self):
        """Check if the component weights are normalized."""
        # precision loss in sum of many small numbers, so can't expect 1.0 exactly
        return _np.allclose(self.weights.sum(), 1.0)

    def prune(self,threshold=0.0):
        """Remove components with weight less or equal ``threshold``.
        Return list of removed components and weights in the form:
        [(index, component, weight), ...].

        :param threshold:

            Float; components with lower weight are deleted

        """
        # go reverse s.t. indices remain valid
        removed_indices    = []
        removed_components = []
        n = len(self.weights)
        for i, c in enumerate(reversed(self.components)):
            if self.weights[n - i - 1] <= threshold:
                current_index = n - i - 1
                removed_indices.append(current_index)

                removed_components.append( (
                         current_index
                        ,self.components.pop(current_index) #this also removes the component
                        ,self.weights[current_index]
                    ) )

        # adjust weights
        self.weights = _np.delete(self.weights, removed_indices)

        return removed_components

    @_add_to_docstring("        " + _msg_expect_normalized_weights)
    @_add_to_docstring(''':param individual:\n
            bool; If true, return the evaluation of each component at ``x`` as
            an array.\n\n''')
    @_inherit_docstring(ProbabilityDensity)
    def evaluate(self, x, individual=False):
        components_evaluated = _np.empty(len(self.components))
        for i,comp in enumerate(self.components):
            components_evaluated[i] = comp.evaluate(x)
        # avoid direct exponentiation --> use logsumexp
        res = logsumexp(components_evaluated, self.weights)
        if individual:
            return res, components_evaluated
        else:
            return res

    def multi_evaluate(self, _np.ndarray[double, ndim=2] x not None, _np.ndarray[double, ndim=1] out=None,
                       _np.ndarray[double, ndim=2] individual=None, components=None):
        '''Evaluate density at all points in ``x`` for all components
        specified in ``components`` and return in ``individual``.
        Return log(q(x)) for each point (if ``components`` is ``None``).

        Use ``individual`` if you need the density at ``x`` for each component
        and you want to compute it only once to increase efficiency.

        :param x:
            (N x D) array; one D-dim. sample per row.

        :param out:
            (N) array, optional; write output into this array (if
            ``components`` is ``None``).

        :param individual:
            (N x K) array; density of k-th component at the n-th sample.

        :param components:
            Iterable of integers; calculate ``individual`` for these
            components only.

        '''
        assert x.shape[1] == self.dim, "The points in ``x`` have the wrong dimension (%i instead of %i)" % (x.shape[1], self.dim)
        if individual is None:
            individual = _np.empty( (len(x), len(self)) )
        else:
            assert len(x) == len(individual), "For the provided ``x``, ``individual`` must have shape %s" % ((len(x), len(self)),)
            assert individual.shape[1] == len(self), "For the provided ``x``, ``individual`` must have shape %s" % ((len(x), len(self)),)

        if components is None:
            for k,c in enumerate(self.components):
                c.multi_evaluate(x, individual[:,k])
            if out is None:
                return logsumexp2D(individual, self.weights)
            else:
                assert len(out) == len(x), '``out`` must have length %i' % (len(x))
                out[:] = logsumexp2D(individual, self.weights)
                return out

        else:
            assert out is None, 'If ``components`` is not None, ``out`` must be None.'
            for k in components:
                self.components[k].multi_evaluate(x, individual[:,k])

    @_add_to_docstring(_msg_expect_normalized_weights)
    def propose(self, N=1, rng=_np.random.mtrand, trace=False, shuffle=True):
        """Propose N points using the random number generator ``rng``.

        :param N:

            Integer; the number of random points to be proposed

        :param rng:

            State of a random number generator like numpy.random.mtrand

        :param shuffle:

            bool; if True (default), the samples are disordered. Otherwise,
            the samples are ordered by the components.

        :param trace:

            bool; if True, return the proposed samples and an array containing
            the number of the component responsible for each sample, otherwise
            just return the samples.

        .. important::
            ``rng`` must:
               - return a numpy array of N samples from the multinomial
                 distribution with probabilities ``pvals`` when calling
                 **rng.multinomial(N, pvals)**
               - shuffle an ``array`` in place when calling **rng.shuffle(array)**

        """
        if trace and shuffle:
            raise ValueError('Either ``shuffle`` or ``trace`` must be ``False``!')

        to_get = rng.multinomial(N, self.weights)

        current_write_start = 0
        output_samples = _np.empty((N,self.dim))

        for i, comp in enumerate(self.components):
            if to_get[i] != 0:
                output_samples[current_write_start:current_write_start + to_get[i]] = comp.propose(to_get[i])
            current_write_start += to_get[i]

        if trace:
            current_write_start = 0
            output_origin = _np.empty(N, dtype=int)
            for i in range(len(self.components)):
                output_origin[current_write_start:current_write_start + to_get[i]] = i
                current_write_start += to_get[i]
            return output_samples, output_origin
        else: # if not trace
            if shuffle:
                rng.shuffle(output_samples)
            return output_samples

def create_gaussian_mixture(means, covs, weights=None):
    """
    Create a :py:class:`.MixtureDensity` with Gaussian (:py:class:`.Gauss`)
    components. The output can be used for the clustering algorithms.

    .. seealso::

        :py:func:`.recover_gaussian_mixture`, :py:func:`.create_t_mixture`,
        :py:func:`.recover_t_mixture`

    :param means:

        Vector-like array; the means of the Gaussian mixture

    :param covs:

        3d-matrix-like array; the covariances of the Gaussian mixture.
        ``cov[i]`` will be interpreted as the i'th covariance matrix.

    :param weights:

        Vector-like array, optional; the component weights. If not
        provided all components will get equal weight.

        .. note::

            The weights will automatically be normalized.

    """
    assert len(means) == len(covs), 'Number of means (%i) does not match number of covariances (%i)' % (len(means), len(covs) )
    components = []
    for mean, cov in zip(means, covs):
        components.append(Gauss(mean, cov))
    return MixtureDensity(components, weights)

def recover_gaussian_mixture(mixture):
    """
    Extract the means, covariances and component weights from a
    :py:class:`.MixtureDensity`.

    .. seealso::

        :py:func:`.create_gaussian_mixture`, :py:func:`.create_t_mixture`,
        :py:func:`.recover_t_mixture`

    :param mixture:

        :py:class:`.MixtureDensity` with Gaussian (:py:class:`.Gauss`)
        components; the mixture to be decomposed.

    """
    weights = _np.array(mixture.weights)

    dim = mixture.dim
    N   = len(weights)

    means = _np.empty((N, dim))
    covs  = _np.empty((N, dim, dim))

    for i, c in enumerate(mixture.components):
        means[i] = c.mu
        covs [i] = c.sigma

    return means, covs, weights

def create_t_mixture(means, covs, dofs, weights=None):
    """
    Create a :py:class:`.MixtureDensity` with Student t
    (:py:class:`.StudentT`) components. The output can be used for the
    clustering algorithms.

    .. seealso::

        :py:func:`.recover_t_mixture`, :py:func:`.create_gaussian_mixture`,
        :py:func:`.recover_gaussian_mixture`

    :param means:

        Vector-like array; the means of the mixture

    :param covs:

        3d-matrix-like array; the covariances of the mixture. ``cov[i]``
        will be interpreted as the i'th covariance matrix.

    :param dofs:

        Vector-like array; the degrees of freedom

    :param weights:

        Vector-like array, optional; the component weights. If not
        provided all components will get equal weight.

        .. note::

            The weights will automatically be normalized.

    """
    assert ( len(means) == len(covs) ) and ( len(means) == len(dofs) ), \
           'Number of ``means`` (%i), ``covs`` (%i) and ``dofs`` (%i) do not match.' % ( len(means), len(covs), len(dofs) )
    components = []
    for mean, cov, dof in zip(means, covs, dofs):
        components.append(StudentT(mean, cov, dof))
    return MixtureDensity(components, weights)

def recover_t_mixture(mixture):
    """
    Extract the means, covariances, degrees of freedom and component
    weights from a :py:class:`.MixtureDensity`.

    .. seealso::

        :py:func:`.create_t_mixture`, :py:func:`.create_gaussian_mixture`,
        :py:func:`.recover_gaussian_mixture`

    :param mixture:

        :py:class:`.MixtureDensity` with Student t (:py:class:`.StudentT`)
        components; the mixture to be decomposed.

    """
    weights = _np.array(mixture.weights)

    dim = mixture.dim
    N   = len(weights)

    means = _np.empty((N, dim))
    covs  = _np.empty((N, dim, dim))
    dofs  = _np.empty((N))

    for i, c in enumerate(mixture.components):
        means[i] = c.mu
        covs [i] = c.sigma
        dofs [i] = c.dof

    return means, covs, dofs, weights
