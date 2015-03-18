"""Collect importance sampler for Population Monte Carlo

"""

import numpy as _np
from math import exp as _exp
from copy import deepcopy as _cp
from ..tools._doc import _inherit_docstring
from ..tools import History as _History
from ..tools.indicator import merge_function_with_indicator as _indmerge

def calculate_expectation(samples, f):
    r'''Calculate the expectation value of function ``f`` using weighted
    samples (like the output of an importance-sampling run).

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
    r'''Calculate the mean of weighted samples (like the output of an
    importance-sampling run).

    :param samples:

        Matrix-like numpy array; the samples to be used. The first column
        is used as (unnormalized) weights.

    '''
    return _np.average(samples[:,1:], axis=0, weights=samples[:,0])

def calculate_covariance(samples):
    r'''Calculates the covariance matrix of weighted samples (like the output of an
    importance-sampling run).

    :param samples:

        Matrix-like numpy array; the samples to be used. The first column
        is used as (unnormalized) weights.

    '''
    sum_weights_sq = (samples[:,0].sum())**2
    sum_sq_weights = (samples[:,0]**2).sum()

    mean  = calculate_mean(samples)

    return sum_weights_sq / (sum_weights_sq - sum_sq_weights)  *\
           calculate_expectation(samples, lambda x: _np.einsum('i,j', x - mean, x - mean))

_docstring_params_importance_sampler = """:param target:

        The target density. Must be a function accepting a 1d numpy
        array and returning a float, namely :math:`\log(P(x))`,
        the log of the target `P`.

    :param proposal:

        The proposal density `q`. Should be of type
        :py:class:`pypmc.density.base.ProbabilityDensity`.

    :param indicator:

        The indicator function receives a numpy array and returns bool.
        The target is only called if indicator(proposed_point) returns
        True. Otherwise, the proposed point will get zero-weight without
        call to target.
        Use this function to specify the support of the target.

        .. seealso::
            :py:mod:`pypmc.tools.indicator`

    :param prealloc:

        An integer, defines the number of Points for which memory in
        ``history`` is allocated. If more memory is needed, it will be
        allocated on demand.

        .. hint::
            Preallocating memory can speed up the calculation, in
            particular if it is known in advance how long the chains
            are run.

    :param save_target_values:

        Bool; if ``True``, store the evaluated ``target`` at every visited
        point in ``self.target_values``

    :param rng:

        The rng passed to the proposal when calling proposal.propose

        .. important::
            ``rng`` must fulfill the requirements of your proposal
            :py:meth:`pypmc.density.base.ProbabilityDensity.propose`

    """

class ImportanceSampler(object):
    __doc__ = r"""An importance sampler object; generates weighted samples from
    ``target`` using ``proposal``.

    """ + _docstring_params_importance_sampler
    def __init__(self, target, proposal, indicator=None, prealloc=0,
                 save_target_values=False, rng=_np.random.mtrand):
        self.proposal = _cp(proposal)
        self.rng      = rng
        self.target   = _indmerge(target, indicator, -_np.inf)
        self.target_values = _History(1, prealloc) if save_target_values else None
        self.history  = _History(proposal.dim + 1, prealloc)

    def clear(self):
        '''Clear history of samples and other internal variables to free memory.

        .. note::
            The proposal is untouched.

        '''
        self.history.clear()
        if self.target_values is not None:
            self.target_values.clear()

    def run(self, N=1, trace_sort=False):
        '''Runs the sampler and stores the history of visited points into
        the member variable ``self.history``

        .. seealso::
            :py:class:`pypmc.tools.History`

        :param N:

            An int which defines the number of steps to run the chain.

        :param trace_sort:

            Bool; if True, return an array containing the responsible
            component of ``self.proposal`` for each sample generated
            during this run.

            .. note::

                This option only works for proposals of type
                :py:class:`pypmc.density.mixture.MixtureDensity`

            .. note::

                If True, the samples will be ordered by the components.

        '''
        if N == 0:
            return 0

        if trace_sort:
            this_run, origin = self._get_samples(N, trace_sort=True)
            self._calculate_weights(this_run, N)
            return origin
        else:
            this_run = self._get_samples(N, trace_sort=False)
            self._calculate_weights(this_run, N)

    def _calculate_weights(self, this_run, N):
        """Calculates and saves the weights of a run."""
        if self.target_values is None:
            for i in range(N):
                tmp = this_run[i, 1:]
                tmp = self.target(tmp) - self.proposal.evaluate(tmp)
                this_run[i,0] = _exp(tmp)
        else:
            this_target_values = self.target_values.append(N)
            for i in range(N):
                tmp = this_run[i, 1:]
                this_target_values[i] = self.target(tmp)
                tmp = this_target_values[i] - self.proposal.evaluate(tmp)
                this_run[i,0] = _exp(tmp)

    def _get_samples(self, N, trace_sort):
        """Saves N samples from ``self.proposal`` to ``self.history``
        Does NOT calculate the weights.

        Returns a reference to the samples in ``self.history``.
        If trace is True, additionally returns an array indicating
        the responsible component. (MixtureDensity only)

        """
        # allocate an empty numpy array to store the run and append accept count
        # (importance sampling accepts all points)
        this_run = self.history.append(N)

        # store the proposed points (weights are still to be calculated)
        if trace_sort:
            this_run[:,1:], origin = self.proposal.propose(N, self.rng, trace=True, shuffle=False)
            return this_run, origin
        else:
            this_run[:,1:] = self.proposal.propose(N, self.rng)
            return this_run

def combine_weights(samples, weights, proposals):
    """Calculate the `deterministic mixture weights` according to
    [Cor+12]_ given ``samples``, standard ``weights`` and their ``proposals`` for a
    number of steps in which importance samples are computed for the same target
    density but different proposals.

    Return the weights as a :class:`pypmc.tools.History` such that the
    weights for each proposal are easily accessible.

    :param samples:

        Iterable of matrix-like arrays; the weighted samples whose importance
        weights shall be combined. One sample per row in each array, one array
        for each step, or different proposal.

    :param weights:

        Iterable of 1D arrays; the standard importance weights
        :math:`P(x_i^t)/q_t(x_i^t)`. Each array in the iterable contains all
        weights of the samples of step ``t``, they array's size has to match the
        ``t``-th entry in samples.

    :param proposals:

        Iterable of :py:class:`pypmc.density.base.ProbabilityDensity` instances;
        the proposal densities from which the ``samples`` have been
        drawn.

    """
    # shallow copy --> can safely modify (need numpy arrays --> can overwrite with np.asarray)
    samples = list(samples)
    weights = list(weights)

    assert len(samples) == len(weights), \
    "Got %i importance-sampling runs but %i weights" % (len(samples), len(weights))

    assert len(samples) == len(proposals), \
    "Got %i importance-sampling runs but %i proposal densities" % (len(samples), len(proposals))

    # number of samples from each proposal
    N = _np.empty(len(proposals))
    N_total = 0

    # basic consistency checks, conversion to numpy array and counting of the total number of samples
    for i in range(len(N)):
        samples[i] = _np.asarray(samples[i])
        assert len(samples[i].shape) == 2, '``samples[%i]`` is not matrix like.' % i
        dim = samples[0].shape[-1]
        assert samples[i].shape[-1] == dim, \
            "Dimension of samples[0] (%i) does not match the dimension of samples[%i] (%i)" \
            % (dim, i, samples[i].shape[-1])
        N[i] = len(samples[i])
        N_total += N[i]

        weights[i] = _np.asarray(weights[i])
        assert N[i] == len(weights[i]), \
            'Length of weights[%i] (%i) does not match length of samples[%i] (%i)' \
            % (i, N[i], i, len(weights[i]))

    combined_weights_history = _History(1, N_total)

    # if all weights positive => prefer log scale
    all_positive = True
    for w in weights:
        all_positive &= (w[:] > 0.0).all()
        if not all_positive:
            break
    if all_positive:
        combined_weights_history = _combine_weights_log(samples, weights, proposals, combined_weights_history, N_total, N)
    else:
        combined_weights_history = _combine_weights_linear(samples, weights, proposals, combined_weights_history, N_total, N)

    assert _np.isfinite(combined_weights_history[:][:,0]).all(), 'Encountered inf or nan mixture weights'
    return combined_weights_history

def _combine_weights_linear(samples, weights, proposals, combined_weights_history, N_total, N):
    # now the actual combination: [Cor+12], Eq. (3)

    # on linear scale
    for t, this_proposal in enumerate(proposals):
        this_combined_weights = combined_weights_history.append(N[t])

        this_combined_weight_denominator = 0.0
        for j, prop in enumerate(proposals):
            this_combined_weight_denominator += N[j] * _np.exp(prop.multi_evaluate(samples[t]))
        this_combined_weight_denominator /= N_total

        this_target_values = _np.exp(this_proposal.multi_evaluate(samples[t])) * weights[t]

        this_combined_weights[:][:,0] = this_target_values / this_combined_weight_denominator

    return combined_weights_history

def _combine_weights_log(samples, weights, proposals, combined_weights_history, N_total, N):
    # on log scale in their notation
    # log w_i^t = log(omega_i^t) + log(q_i^t) + log(\sum_j N_j) - log(\sum_l N_l exp(log(q_l(y_i^t))))
    # where omega is the ordinary importance weight
    for t, this_proposal in enumerate(proposals):
        # "subarray" for this step t, part of big array of all mixture weights
        combined_weights = combined_weights_history.append(N[t])

        # actually collection of vectors y^t_i for all i
        y = samples[t]

        # evaluate proposal at step t for all the samples
        log_q_t = this_proposal.multi_evaluate(y)

        # mixture weights on log scale: assume w>0!
        log_w_t = _np.log(weights[t]).copy()
        log_w_t += log_q_t
        log_w_t += _np.log(N_total)

        # matrix of all proposal evaluated at every sample in step t
        q = _np.empty((N[t], len(proposals)))
        q[:,t] = log_q_t

        # loop over all indices l != t
        other_steps = list(range(len(proposals)))
        other_steps.pop(t)
        for l in other_steps:
            q[:,l] = proposals[l].multi_evaluate(y)

        # use logsumexp in case q are so small that exp(q)=0 in double precision
        from ..tools._regularize import logsumexp2D
        log_w_t -= logsumexp2D(q, N)

        # return to linear scale
        combined_weights[:][:,0] = _np.exp(log_w_t)

    # return mixture weights for ALL steps
    sum_w = combined_weights_history[:][:,0].sum()
    assert sum_w > 0, 'Sum of weights <=0 (%g)' % sum_w

    return combined_weights_history
