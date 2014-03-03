"""Collect probability densities"""

import numpy as _np
from copy import deepcopy as _deepcopy
from scipy.misc import logsumexp as _lse
from ..markov_chain import proposal as _mc_proposal
from ..tools._doc import _inherit_docstring, _add_to_docstring

class ProbabilityDensity(object):
    """Abstract Base class of a probability density. Can be used as proposal
    for the importance sampler.

    """
    dim = 0

    def __init__(self):
        raise NotImplementedError('Do not create instances from this class, use derived classes instead.')

    def evaluate(self, x):
        """Evaluate log of the density to propose ``x``, namely log(q(x)).

        :param x:

            Vector-like numpy-array; the proposed point.

        """
        raise NotImplementedError()

    def propose(self, N=1, rng=_np.random.mtrand):
        """propose(self, N=1, rng=numpy.random.mtrand)

        Propose N points using the random number generator ``rng``.

        :param N:

            Integer; the number of random points to be proposed

        :param rng:

            The state of a random number generator like numpy.random.mtrand

        """
        raise NotImplementedError()

_msg_expect_normalized_weights = \
    """.. important::

            This function assumes that the weights are normalized, i.e.
            that ``self.normalized()`` returns ``True``.

    """

class MixtureDensity(ProbabilityDensity):
    """Base class for multimodal probability densities.

    :param components:

        Iterable of ProbabilityDensities; the mixture's components

    :param weights:

        Iterable of floats; the weights of the components
        (will be normalized automatically during initialization)

    Slice and item access is supported:
    ``self[item]`` returns ``tuple(self.components[item],self.weights[item])``

    Iteration is supported:
    Step ``i`` yields ``tuple(self.components[i],self.weights[i])``

    """
    def __init__(self, components, weights=None):
        self.components  = [_deepcopy(component) for component in components]

        self.dim = self.components[0].dim

        _np.testing.assert_equal([comp.dim for comp in self.components],[self.dim for comp in components])

        if weights is None:
            self.weights = _np.ones(len(self.components))
        else:
            self.weights = _np.array(weights, dtype=float)
            assert len(self.weights) == len(self.components)

        self.normalize()

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

    @_add_to_docstring(_msg_expect_normalized_weights)
    @_inherit_docstring(ProbabilityDensity)
    def evaluate(self, x):
        components_evaluated = _np.empty(len(self.components))
        for i,comp in enumerate(self.components):
            components_evaluated[i] = comp.evaluate(x)
        # avoid direct exponentiation --> use scipy.misc.logsumexp (_lse)
        return _lse(a=components_evaluated, b=self.weights)

    @_add_to_docstring(_msg_expect_normalized_weights)
    @_add_to_docstring(""":param shuffle:\n
            bool; if True (default), the samples are disordered. Otherwise,
            the samples are ordered by the components.\n\n        """)
    @_add_to_docstring(""":param trace:\n
            bool; if True, return the proposed samples and an array containing
            the number of the component responsible for each sample, otherwise
            just return the samples.\n\n        """)
    @_add_to_docstring("""    .. important::\n
                ``rng`` must return a numpy array of N samples from the multinomial
                distribution with probabilities `pvals` when calling
                ``rng.multinomial(N, pvals)``.\n\n\n        """)
    @_add_to_docstring(ProbabilityDensity.propose.__doc__.replace('.mtrand)', '.mtrand, trace=False)', 1))
    def propose(self, N=1, rng=_np.random.mtrand, trace=False, shuffle=True):
        ""
        if trace and shuffle:
            raise ValueError('Either ``shuffle`` or ``trace`` must be ``False``!')

        to_get = rng.multinomial(N, self.weights)

        current_write_start = 0
        output_samples = _np.empty((N,self.dim))

        #TODO: parallelize the sampling from components
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
                _np.random.shuffle(output_samples)
            return output_samples

    def __getitem__(self, i):
        return self.components[i] , self.weights[i]

    def __iter__(self):
        for i in range(len(self.components)):
            yield self.components[i],self.weights[i]

class Gauss(ProbabilityDensity):
    r"""A Gaussian probability density. Can be used as component for
    MixtureDensities.

        :param mu:

            Vector-like numpy-array; the gaussian's mean :math:`\mu`

        :param sigma:

            Matrix-like numpy-array; the gaussian's covariance matrix :math:`\Sigma`

    """
    def __init__(self, mu, sigma):
        self.update(mu, sigma)

    def update(self, mu, sigma):
        r"""Re-initiate the density with new mean and covariance matrix.

        :param mu:

            Vector-like numpy-array; the gaussian's mean :math:`\mu`

        :param sigma:

            Matrix-like numpy-array; the gaussian's covariance matrix
            :math:`\Sigma`

        """
        self.mu          = _np.array(mu)
        self.dim         = len(self.mu)

        self.mc_proposal = _mc_proposal.MultivariateGaussian(sigma)

        self.inv_sigma   = self.mc_proposal.inv_sigma # creates reference
        self.det_sigma   = self.mc_proposal.det_sigma # creates copy because det_sigma is a float
        self.sigma       = self.mc_proposal.sigma     # creates reference

        assert self.dim == self.sigma.shape[0], "Dimensions of mean (%d) and covariance matrix (%d) do not match!" %(self.dim,self.sigma.shape[0])

    @_inherit_docstring(ProbabilityDensity)
    def evaluate(self, x):
        return self.mc_proposal.evaluate(x,self.mu)

    @_add_to_docstring("""    .. important::\n
                ``rng`` must meet the requirements of
                :py:meth:`pypmc.markov_chain.proposal.MultivariateGaussian.propose`.\n\n""")
    @_inherit_docstring(ProbabilityDensity)
    def propose(self, N=1, rng = _np.random.mtrand):
        output = _np.empty((N,self.dim))
        for i in range(N):
            output[i] = self.mc_proposal.propose(self.mu)
        return output

class StudentT(ProbabilityDensity):
    r"""A Student T probability density. Can be used as component for
    MixtureDensities.

        :param mu:

            Vector-like numpy-array; the gaussian's mean :math:`\mu`

        :param sigma:

            Matrix-like numpy-array; the gaussian's covariance matrix :math:`\Sigma`

        :param dof:

            Float; the degrees of freedom :math:`\nu`

    """
    def __init__(self, mu, sigma, dof):
        self.update(mu, sigma, dof)

    def update(self, mu, sigma, dof):
        r"""Re-initiate the density with new mean, covariance matrix and
        degrees of freedom.

        :param mu:

            Vector-like numpy-array; the gaussian's mean :math:`\mu`

        :param sigma:

            Matrix-like numpy-array; the gaussian's covariance matrix
            :math:`\Sigma`

        :param dof:

            Float; the degrees of freedom :math:`\nu`

        """
        self.mu          = _np.array(mu)
        self.dim         = len(self.mu)
        self.dof         = dof

        self.mc_proposal = _mc_proposal.MultivariateStudentT(sigma, dof)

        self.inv_sigma   = self.mc_proposal.inv_sigma
        self.det_sigma   = self.mc_proposal.det_sigma
        self.sigma       = self.mc_proposal.sigma

        assert self.dim == self.sigma.shape[0], "Dimensions of mean (%d) and covariance matrix (%d) do not match!" %(self.dim,self.sigma.shape[0])

    @_inherit_docstring(ProbabilityDensity)
    def evaluate(self, x):
        return self.mc_proposal.evaluate(x,self.mu)

    @_add_to_docstring("""    .. important::\n
                ``rng`` must meet the requirements of
                :py:meth:`pypmc.markov_chain.proposal.MultivariateStudentT.propose`.\n\n""")
    @_inherit_docstring(ProbabilityDensity)
    def propose(self, N=1, rng=_np.random.mtrand):
        output = _np.empty((N,self.dim))
        for i in range(N):
            output[i] = self.mc_proposal.propose(self.mu)
        return output
