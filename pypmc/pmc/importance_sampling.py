"""Collect importance sampler for Population Monte Carlo

"""

import numpy as _np
from math import exp
from copy import deepcopy as _cp
from .._tools._doc import _inherit_docstring
from .._tools._chain import _Chain, _merge_function_with_indicator

def calculate_expectation(samples, f):
    r'''Calculates the expectation value of function ``f`` using weighted
    samples (like the output of a PMC-run).

    Denoting :math:`x_n` as the sample n and :math:`w_n` as its (normalized)
    weight, the following is returned:

    .. math::

        \sum_{n=1}^{N} w_n f(x_n)
        \mathrm{\ \ where\ \ } \sum_{n=1}^{N}w_n \overset{!}{=} 1

    :param samples:

        Matrix-like numpy array; the samples to be used. The first column
        is used as (unnormalized) weights.

    :param f:

        Callable, the function to be evaluated.

    '''
    normalization = 0.
    out           = 0.
    for point in samples:
        normalization += point[0]
        out += point[0] * f(point[1:])
    return out/normalization

def calculate_mean(samples):
    r'''Calculates the mean of weighted samples (like the output of a
    PMC-run).

    :param samples:

        Matrix-like numpy array; the samples to be used. The first column
        is used as (unnormalized) weights.

    '''
    return _np.average(samples[:,1:], axis=0, weights=samples[:,0])

def calculate_covariance(samples):
    r'''Calculates the covariance matrix of weighted samples (like the
    output of a PMC-run).

    :param samples:

        Matrix-like numpy array; the samples to be used. The first column
        is used as (unnormalized) weights.

    '''
    sum_weights_sq = (samples[:,0].sum())**2
    sum_sq_weights = (samples[:,0]**2).sum()

    mean  = calculate_mean(samples)

    return sum_weights_sq / (sum_weights_sq - sum_sq_weights)  *\
           calculate_expectation(samples, lambda x: _np.einsum('i,j', x - mean, x - mean))

class ImportanceSampler(_Chain):
    r"""ImportanceSampler(target, proposal, indicator = None, prealloc = 0,
    rng = numpy.random.mtrand)

    An importance sampler object; generates weighted samples from
    ``target`` using ``proposal``.

    :param target:

        The target density. Must be a function accepting a 1d numpy
        array and returning a float, namely :math:`\log(P(x))`,
        the log of the target `P`.

    :param proposal:

        The proposal density `q`.
        Should be of type :py:class:`pypmc.pmc.proposal.PmcProposal`.

    :param indicator:

        The indicator function receives a numpy array and returns bool.
        The target is only called if indicator(proposed_point) returns
        True. Otherwise, the proposed point will get zero-weight without
        call to target.
        Use this function to specify the support of the target.

        .. seealso::
            :py:mod:`pypmc.indicator_factory`

    :param prealloc:

        An integer, defines the number of Markov chain points for
        which memory in ``hist`` is allocated. If more memory is
        needed, it will be allocated on demand.

        .. hint::
            Preallocating memory can speed up the calculation, in
            particular if it is known in advance how long the chains
            are run.

    :param rng:

        The rng passed to the proposal when calling proposal.propose

        .. important::
            ``rng`` must fulfill the requirements of your proposal
            :py:meth:`pypmc.pmc.proposal.PmcProposal.propose`

    """
    def __init__(self, target, proposal, indicator = None, prealloc = 0, rng = _np.random.mtrand):
        self.proposal  = _cp(proposal)
        self.rng       = rng
        self.target    = _merge_function_with_indicator(target, indicator, -_np.inf)

        # need to draw one weighted sample to initialize the history
        point  = proposal.propose()[0]
        weight = exp(self.target(point) - proposal.evaluate(point))
        start  = _np.hstack( (weight, point) )

        super(ImportanceSampler, self).__init__(start = start, prealloc = prealloc)
        del self.current

    @_inherit_docstring(_Chain)
    def run(self, N = 1):
        # allocate an empty numpy array to store the run and append accept count
        # (importance sampling accepts all points)
        this_run = self.hist._alloc(N)
        self.hist._append_accept_count(N)

        # store the proposed points (weights are still to be calculated)
        this_run[:,1:] = self.proposal.propose(N, self.rng)

        # calculate and save the weights
        for i in range(N):
            tmp = this_run[i, 1:]
            tmp = self.target(tmp) - self.proposal.evaluate(tmp)
            this_run[i,0] = exp(tmp)
