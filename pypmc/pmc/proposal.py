"""Collect PMC proposal densities"""

import numpy as _np
from copy import deepcopy as _deepcopy
from math import exp, log
from ..markov_chain import proposal as _mc_proposal
from .._tools._doc import _inherit_docstring, _add_to_docstring

class PmcProposal(object):
    """Abstract Base class for a proposal density for the Population
    Monte Carlo sampler.

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

#TODO: find a way to combine this and pypmc.cluster.gaussian_mixture

_msg_expect_normalized_weights = \
    """.. important::

            This function assumes that the weights are normalized, i.e.
            that ``self.normalized()`` returns ``True``.

    """

class MixtureProposal(PmcProposal):
    """Base class for multimodal proposal densities.

    :param components:

        Iterable of PmcProposals; the Proposal components

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
        Return list of removed components and weights.

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
                         self.components.pop(current_index) #this also deletes the component
                        ,self.weights[current_index]
                    ) )

        # adjust weights
        self.weights = _np.delete(self.weights, removed_indices)

        return removed_components

    @_add_to_docstring(_msg_expect_normalized_weights)
    @_inherit_docstring(PmcProposal)
    def evaluate(self, x):
        out = 0.
        for i,weight in enumerate(self.weights):
            out += weight * exp(self.components[i].evaluate(x))
        if out == 0.:
            return -_np.inf
        else:
            return log(out)

    @_add_to_docstring("""    .. warning::\n
            The returned samples are ordered by components. When disordered
            samples are needed use ``numpy.random.shuffle``\n\n""")
    @_add_to_docstring(_msg_expect_normalized_weights)
    @_add_to_docstring(""":param trace:\n
            bool; if True, return the proposed samples and an array containing
            the number of the component responsible for each sample, otherwise
            just return the samples.\n\n        """)
    @_add_to_docstring("""    .. important::\n
                ``rng`` must return a numpy array of N samples from the
                uniform distribution over [0,1) when calling ``rng.rand(N)``.\n\n\n        """)
    @_add_to_docstring(PmcProposal.propose.__doc__.replace('.mtrand)', '.mtrand, trace=False)', 1))
    def propose(self, N=1, rng=_np.random.mtrand, trace=False):
        ""
        # The Algorithm:
        # 1. Draw N samples from standard uniform distribution
        # 2. The first weight is the probability to get a sample from
        #    the first component, i.e. take as many samples from the
        #    first component as there are random numbers < first weight.
        #    Equivalently count how often (random numbers - first weight) < 0
        # 3. For the next component count how often
        #    a) random numbers >= first weight
        #    and
        #    b) random numbers < second weight
        #    For b), as before, count how often (random numbers - second weight) < 0
        #    For a) substract the number of samples which already counted in step 2
        # 4. Subsequently count how often (random numbers - weight i) < 0
        #    and substract the number of samples which have already been counted
        #    for a previous component

        samples    = rng.rand(N)
        to_get     = _np.empty(len(self.weights), dtype=int)
        to_get[-1] = 0 # for first step in for loop
        previous   = 0

        for i, weight in enumerate(self.weights):
            samples  -= weight
            total     = (samples < 0.).sum()
            to_get[i] = total - previous
            previous += total - previous

        assert total == N, 'Are the weights normalized?'

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
            return output_samples

    def __getitem__(self, i):
        return self.components[i] , self.weights[i]

    def __iter__(self):
        for i in range(len(self.components)):
            yield self.components[i],self.weights[i]

class GaussianComponent(PmcProposal):
    r"""A Gaussian component for MixtureProposals

        :param mu:

            Vector-like numpy-array; the gaussian's mean :math:`\mu`

        :param sigma:

            Matrix-like numpy-array; the gaussian's covariance matrix :math:`\Sigma`

    """
    def __init__(self, mu, sigma):
        self.update(mu, sigma)

    def update(self, mu, sigma):
        r"""Re-initiate the proposal with new mean and covariance matrix.

        :param mu:

            Vector-like numpy-array; the gaussian's mean :math:`\mu`

        :param sigma:

            Matrix-like numpy-array; the gaussian's covariance matrix
            :math:`\Sigma`

        """
        self.mu          = _np.array(mu)
        self.sigma       = _np.array(sigma)

        self.dim         = len(self.mu)
        assert self.dim == self.sigma.shape[0], "Dimensions of mean (%d) and covariance matrix (%d) do not match!" %(self.dim,self.sigma.shape[0])

        self.mc_proposal = _mc_proposal.MultivariateGaussian(self.sigma)

    @_inherit_docstring(PmcProposal)
    def evaluate(self, x):
        return self.mc_proposal.evaluate(x,self.mu)

    @_add_to_docstring("""    .. important::\n
                ``rng`` must meet the requirements of
                :py:meth:`pypmc.markov_chain.proposal.MultivariateGaussian.propose`.\n\n""")
    @_inherit_docstring(PmcProposal)
    def propose(self, N=1, rng = _np.random.mtrand):
        output = _np.empty((N,self.dim))
        for i in range(N):
            output[i] = self.mc_proposal.propose(self.mu)
        return output

class StudentTComponent(PmcProposal):
    r"""A Student T component for MixtureProposals

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
        r"""Re-initiate the proposal with new mean, covariance matrix and
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
        self.sigma       = _np.array(sigma)
        self.dof         = dof

        self.dim         = len(self.mu)
        assert self.dim == self.sigma.shape[0], "Dimensions of mean (%d) and covariance matrix (%d) do not match!" %(self.dim,self.sigma.shape[0])

        self.mc_proposal = _mc_proposal.MultivariateStudentT(self.sigma, self.dof)

    @_inherit_docstring(PmcProposal)
    def evaluate(self, x):
        return self.mc_proposal.evaluate(x,self.mu)

    @_add_to_docstring("""    .. important::\n
                ``rng`` must meet the requirements of
                :py:meth:`pypmc.markov_chain.proposal.MultivariateStudentT.propose`.\n\n""")
    @_inherit_docstring(PmcProposal)
    def propose(self, N=1, rng=_np.random.mtrand):
        output = _np.empty((N,self.dim))
        for i in range(N):
            output[i] = self.mc_proposal.propose(self.mu)
        return output
