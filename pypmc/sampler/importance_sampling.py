"""Collect importance sampler for Population Monte Carlo

"""

import numpy as _np
from math import exp as _exp
from copy import deepcopy as _cp
from ..tools._doc import _inherit_docstring
from ..tools import History as _History
from ..tools.indicator import merge_function_with_indicator as _indmerge

def calculate_expectation(samples, f):
    r'''Calculates the expectation value of function ``f`` using weighted
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
    r'''Calculates the mean of weighted samples (like the output of an
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
        self.save_target_values  = bool(save_target_values)

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
        if trace_sort:
            this_run, origin = self._get_samples(N, trace_sort=True)
            self._calculate_weights(this_run, N)
            return origin
        else:
            this_run = self._get_samples(N, trace_sort=False)
            self._calculate_weights(this_run, N)

    def _calculate_weights(self, this_run, N):
        """Calculates and saves the weights of a run."""
        if not self.save_target_values:
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

class DeterministicIS(ImportanceSampler):
    __doc__ = r"""An importance sampler object; generates weighted samples from
    ``target`` using ``proposal``. Calculates `deterministic mixture
    weights` according to [Cor+12]_ and optionally standard weights.

    """ + _docstring_params_importance_sampler + \
    """:param std_weights:

        Bool; if True, store standard weights in ``self.std_weights``

        .. note::
            Can only be passed as keyword argument.

    """
    def __init__(self, *args, **kwargs):
        save_std_weights = kwargs.pop('std_weights', False)

        super(DeterministicIS, self).__init__(*args, **kwargs)

        # optionally save standars weights
        if save_std_weights:
            self.std_weights = _History(1, self.history.prealloc)

        # save all past proposals in this list
        self.proposal_history = []

        # save all evaluated target and proposal values in this History object
        self._deltas_targets_evaluated = _History(2, self.history.prealloc)

    @_inherit_docstring(ImportanceSampler)
    def clear(self):
        super(DeterministicIS, self).clear()
        self._deltas_targets_evaluated.clear()
        self.proposal_history = []
        try:
            self.std_weights.clear()
        except AttributeError:
            pass

    @_inherit_docstring(ImportanceSampler)
    def _calculate_weights(self, this_weights_samples, this_N):
        inconsistency_message = 'Inconsistent state encountered. If you used ' + \
                                '``self.history.clear()`` try ``self.clear()`` instead.'

        # append proposal for this run to history
        self.proposal_history.append(_cp(self.proposal))

        # allocate memory for new target and proposal evaluations
        this_deltas_targets = self._deltas_targets_evaluated.append(this_N)

        # allocate memory for ``self.target_vales`` (target on log scale)
        if self.target_values is not None:
            this_log_target_values = self.target_values.append(this_N)[:,0]

        # allocate memory for new standard weights (if desired by user)
        try:
            this_std_weights = self.std_weights.append(this_N)[:,0]
            need_std_weights = True
        except AttributeError:
            need_std_weights = False

        # create references
        this_samples = this_weights_samples[:,1:]
        this_deltas  = this_deltas_targets [:,0 ]
        this_targets = this_deltas_targets [:,1 ]
        old_weights_samples = self.history[:-1]
        old_deltas_targets  = self._deltas_targets_evaluated[:-1]
        assert len(old_weights_samples) == len(old_deltas_targets), inconsistency_message

        all_weights_samples = self.history[:]
        all_deltas_targets  = self._deltas_targets_evaluated[:]
        assert len(all_weights_samples) == len(all_deltas_targets), inconsistency_message

        all_weights = all_weights_samples[:,0 ]
        all_samples = all_weights_samples[:,1:]
        all_deltas  = all_deltas_targets [:,0 ]
        all_targets = all_deltas_targets [:,1 ]

        if need_std_weights:
            # calculate the deltas and standard weights for the new samples
            this_deltas[:] = 0.
            i_run = -1
            for i_run in range(len(self.history) - 1): # special treatment for last run
                this_deltas[:] += len(self.history[i_run]) * _np.exp( self.proposal_history[i_run].multi_evaluate(this_samples) )
            # last run
            i_run += 1
            this_std_weights[:] = - self.proposal_history[i_run].multi_evaluate(this_samples)
            this_deltas[:] += len(self.history[i_run]) * _np.exp(- this_std_weights)

            # evaluate the target at the new samples
            for i, sample in enumerate(this_samples):
                tmp = self.target(sample)
                this_std_weights[i] += tmp
                if self.target_values is not None:
                    this_log_target_values[i] = tmp
                # exp because the self.target returns the log of the target
                this_targets[i] = _exp(tmp)
                this_std_weights[i] = _exp(this_std_weights[i])

        else:
            # evaluate the target at the new samples
            for i, sample in enumerate(this_samples):
                tmp = self.target(sample)
                if self.target_values is not None:
                    this_log_target_values[i] = tmp
                # exp because the self.target returns the log of the target
                this_targets[i] = _exp(tmp)

            # calculate the deltas for the new samples
            this_deltas[:] = 0.
            for i_run, i_samples in enumerate(self.history):
                this_deltas[:] += len(i_samples) * _np.exp( self.proposal_history[i_run].multi_evaluate(this_samples) )

        assert i_run + 1 == len(self.proposal_history), inconsistency_message

        # not to be done if this is the first run
        if old_weights_samples.size:
            old_samples = old_weights_samples[:,1:]
            old_deltas  = old_deltas_targets [:,0 ]

            # calculate the deltas for the old samples
            old_deltas[:] += this_N * _np.exp( self.proposal_history[-1].multi_evaluate(old_samples) )

        # calculate the weights (Algorithm1 in [Cor+12])
        all_weights[:] = all_targets / (all_deltas / len(all_weights_samples))

def combine_weights(weighted_samples, proposals):
    """Calculate the `deterministic mixture weights` according to
    [Cor+12]_ given weighted samples from the standard importance
    sampler :py:class:`.ImportanceSampler` and their proposal
    densities.

    :param weighted_samples:

        Iterable of matrix-like arrays; the weighted samples whose
        importance weights shall be combined. The first column of each
        array is interpreted as the weight (on the linear scale!), the
        rest of each row as the sample.  The output of multiple runs
        using :py:meth:`.ImportanceSampler.run()` with different
        proposal densities qualifies as input here.

    :param proposals:

        Iterable of :py:class:`pypmc.density.base.ProbabilityDensity` instances;
        the proposal densities from which the ``weighted_samples`` have been
        drawn.

    """
    # shallow copy --> can safely modify (need numpy arrays --> can overwrite with np.asarray)
    weighted_samples = list(weighted_samples)

    assert len(weighted_samples) == len(proposals), \
    "Got %i importance-sampling runs but %i proposal densities" % (len(weighted_samples), len(proposals))

    # number of samples from each proposal
    N = _np.empty(len(proposals))
    N_total = 0

    # basic consistency checks, conversion to numpy array and counting total number of samples
    for i in range(len(weighted_samples)):
        weighted_samples[i] = _np.asarray(weighted_samples[i])
        assert len(weighted_samples[i].shape) == 2, '``weighted_samples[%i]`` is not matrix like.' % i
        dim = weighted_samples[0].shape[-1] - 1
        assert weighted_samples[i].shape[-1] - 1 == dim, \
            "Dimension of weighted_samples[0] (%i) does not match the dimension of weighted_samples[%i] (%i)" \
                % (dim, i, weighted_samples[i].shape[-1] - 1)
        N[i] = len(weighted_samples[i])
        N_total += N[i]

    combined_weights_history = _History(1, N_total)

    # if all weights positive => prefer log scale
    all_positive = True
    for w in weighted_samples:
        all_positive &= (w[:,0] >= 0.0).all()
        if not all_positive:
            return _combine_weights_linear(weighted_samples, proposals, combined_weights_history, N_total)
    return _combine_weights_log(weighted_samples, proposals, combined_weights_history, N_total, N)

def _combine_weights_linear(weighted_samples, proposals, combined_weights_history, N_total):
    # now the actual combination: [Cor+12], Eq. (3)

    # on linear scale
    for i, (w_x, this_proposal) in enumerate(zip(weighted_samples, proposals)):
        this_combined_weights = combined_weights_history.append(len(w_x))
        this_weights = w_x[:,0 ]
        this_samples = w_x[:,1:]

        this_combined_weight_denominator = 0.0
        for prop, samples_from_prop in zip(proposals, weighted_samples):
            this_combined_weight_denominator += len(samples_from_prop) * _np.exp(prop.multi_evaluate(this_samples))
        this_combined_weight_denominator /= N_total

        this_target_values = _np.exp(this_proposal.multi_evaluate(this_samples)) * this_weights

        this_combined_weights[:][:,0] = this_target_values / this_combined_weight_denominator

    return combined_weights_history[:][:,0]

def _combine_weights_log(weighted_samples, proposals, combined_weights_history, N_total, N):
    # on log scale in their notation
    # log w_i^t = log(omega_i^t) + log(q_i^t) + log(\sum_j N_j) - log(\sum_l N_l exp(log(q_l(y_i^t))))
    # where omega is the ordinary importance weight
    for t, (w_x, this_proposal) in enumerate(zip(weighted_samples, proposals)):
        # "subarray" for this step t, part of big array of all mixture weights
        combined_weights = combined_weights_history.append(len(w_x))
        # these are on the linear scale
        ordinary_weights = w_x[:,0]
        # actually collection of vectors y^t_i for all i
        y = w_x[:,1:]

        # evaluate proposal at step t for all the samples
        log_q_t = this_proposal.multi_evaluate(y)

        # mixture weights on log scale
        log_w_t = _np.log(ordinary_weights).copy()
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
    return combined_weights_history[:][:,0]
