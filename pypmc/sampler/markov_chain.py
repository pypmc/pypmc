"""Collect Markov Chain"""

from copy import deepcopy as _cp
import numpy as _np
from ..tools import History as _History
from ..tools.indicator import merge_function_with_indicator as _indmerge
from ..tools._doc import _inherit_docstring

class MarkovChain(object):
    r"""A Markov chain to generate samples from the target density.

    :param target:

        The target density. Must be a function accepting a 1d numpy
        array and returning a float, namely :math:`\log(P(x))`,
        the log of the target `P`.

    :param proposal:

        The proposal density `q`.
        Should be of type :py:class:`pypmc.density.base.LocalDensity`.

        .. hint::
            If your proposal density is symmetric, define the member
            variable ``proposal.symmetric = True``. This will omit calls
            to proposal.evaluate in the Metropolis-Hastings steps.

    :param start:

        The starting point of the Markov chain. (numpy array)

    :param indicator:

        The indicator function receives a numpy array and returns bool.
        The target is only called if indicator(proposed_point)
        returns True, otherwise the proposed point is rejected
        without call to target.
        Use this function to specify the support of the target.

        .. seealso::
            :py:mod:`pypmc.tools.indicator`

    :param prealloc:

        Integer; the number of Markov chain samples for which memory in
        ``self.samples`` is allocated. If more memory is needed, it will
        be allocated on demand.

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
            ``rng`` must return a sample from the uniform distribution
            in [0,1) when calling **rng.rand()**

        .. seealso::
            ``rng`` must also fulfill the requirements of your proposal
            :py:meth:`pypmc.density.base.LocalDensity.propose`

    """
    def __init__(self, target, proposal, start, indicator=None,
                 prealloc=0, save_target_values=False, rng=_np.random.mtrand):
        # store input into instance
        self.current_point        = _np.array(start, dtype=float) # call array constructor to make sure to have a copy
        self.samples              = _History(len(self.current_point), prealloc)
        self.proposal             = _cp(proposal)
        self.rng                  = rng
        self.target               = _indmerge(target, indicator, -_np.inf)
        self.target_values = _History(1, prealloc) if save_target_values else None
        self.current_target_eval  = self.target(self.current_point)
        if not _np.isfinite(self.current_target_eval):
            raise ValueError('``target(start)`` must evaluate to a finite value and ``indicator(start)`` must be ``True``')

    def clear(self):
        '''Clear history of visited points (stored in ``self.samples``) and
        other internal variables to free memory.

        .. note::
            The current state that defines the Markov chain is untouched.

        '''
        self.samples.clear()
        if self.target_values is not None:
            self.target_values.clear()

    def run(self, N=1, continue_on_NaN=False):
        '''Run the chain and store the history of visited points into
        the member variable ``self.samples``. Returns the number of
        accepted points during the run.

        .. seealso::
            :py:class:`pypmc.tools.History`

        :param N:

            An int which defines the number of steps to run the chain.

        :param continue_on_NaN:

            A boolean flag defining the behavior when encountering an NaN in
            the user-supplied target density for a proposed point.
            Default: ``False`` (-> raise ``ValueError``). If ``True``, reject
            the proposed point and continue.
        '''
        if N == 0:
            return 0

        # set the accept function
        if self.proposal.symmetric:
            get_log_rho = self._get_log_rho_metropolis
        else:
            get_log_rho = self._get_log_rho_metropolis_hastings

        # allocate an empty numpy array to store the run
        if self.target_values is not None:
            this_target_values = self.target_values.append(N)
        this_run     = self.samples.append(N)
        accept_count = 0

        for i_N in range(N):
            # propose new point
            proposed_point = self.proposal.propose(self.current_point, self.rng)
            proposed_eval  = self.target(proposed_point)

            # log_rho := log(probability to accept point), where log_rho > 0 is meant to imply rho = 1
            log_rho = get_log_rho(proposed_point, proposed_eval)

            # check for NaN
            if _np.isnan(log_rho):
                if not continue_on_NaN: raise ValueError('encountered NaN')
                # otherwise reject
                this_run[i_N] = self.current_point
            else:
                # accept if rho >= 1 or with with probability rho (if rho < 1)
                if log_rho >=0 or log_rho >= _np.log(self.rng.rand()) :
                    accept_count += 1
                    this_run[i_N]            = proposed_point
                    self.current_point       = proposed_point
                    self.current_target_eval = proposed_eval

                # reject if not accepted
                else:
                    this_run[i_N] = self.current_point

            # save target value if desired
            if self.target_values is not None:
                this_target_values[i_N] = self.current_target_eval

        # ---------------------- end for --------------------------------

        return accept_count

    def _get_log_rho_metropolis(self, proposed_point, proposed_eval):
        """calculate the log of the metropolis ratio"""
        return proposed_eval - self.current_target_eval

    def _get_log_rho_metropolis_hastings(self, proposed_point, proposed_eval):
        """calculate log(metropolis ratio times hastings factor)"""
        return self._get_log_rho_metropolis(proposed_point, proposed_eval)\
             - self.proposal.evaluate      (proposed_point, self.current) \
             + self.proposal.evaluate      (self.current, proposed_point)

class AdaptiveMarkovChain(MarkovChain):
    # set the docstring --> inherit from Base class, but replace:
    # - MarkovChain(*args, **kwargs) --> AdaptiveMarkovChain(*args, **kwargs)
    # - A Markov chain --> A Markov chain with proposal covariance adaptation
    # - ProposalDensity by Multivariate in description of :param propoasal:
    __doc__ = MarkovChain.__doc__\
    .replace('MarkovChain(', 'AdaptiveMarkovChain(')\
    .replace('A Markov chain', '''A Markov chain with proposal covariance adaptation as in [HST01]_''' , 1)\
    .replace('ProposalDensity', 'Multivariate')

    def __init__(self, *args, **kwargs):
        # set adaptation params
        self.adapt_count = 1

        self.covar_scale_multiplier = kwargs.pop('covar_scale_multiplier' ,   1.5   )

        self.covar_scale_factor     = kwargs.pop('covar_scale_factor'     , None    )
        self.covar_scale_factor_max = kwargs.pop('covar_scale_factor_max' , 100.    )
        self.covar_scale_factor_min = kwargs.pop('covar_scale_factor_min' ,    .0001)

        self.force_acceptance_max   = kwargs.pop('force_acceptance_max'   ,    .35  )
        self.force_acceptance_min   = kwargs.pop('force_acceptance_min'   ,    .15  )

        self.damping                = kwargs.pop('damping'                ,    .5   )

        super(AdaptiveMarkovChain, self).__init__(*args, **kwargs)

        if self.covar_scale_factor is None:
            self.covar_scale_factor = 2.38**2/len(self.current_point)

        # initialize unscaled sigma
        self.unscaled_sigma = self.proposal.sigma / self.covar_scale_factor

    @_inherit_docstring(MarkovChain)
    def run(self, N=1, continue_on_NaN=False):
        if N == 0:
            return 0
        self._last_accept_count = super(AdaptiveMarkovChain, self).run(N, continue_on_NaN)
        return self._last_accept_count

    def set_adapt_params(self, *args, **kwargs):
        r"""Sets variables for covariance adaptation.

        When :meth:`.adapt` is called, the proposal's covariance matrix is
        adapted in order to improve the chain's performance. The aim
        is to improve the efficiency of the chain by making better
        proposals and forcing the acceptance rate :math:`\alpha` of
        the chain to lie in an interval ensuring good exploration:

        :param force_acceptance_max:

            Float, the upper limit (in (0,1])

            Default: :math:`\alpha_{max}=.35`


        :param force_acceptance_min:

            Float, the lower limit (in [0,1))

            Default: :math:`\alpha_{min}=.15`


        This is achieved in two steps:

        1. **Estimate the target covariance**: compute the sample
        covariance from the last (the t-th) run as :math:`S^t`
        then combine with previous estimate :math:`\Sigma^{t-1}`
        with a weight damping out over time as

        .. math::

            \Sigma^t = (1-a^t) \Sigma^{t-1} + a^t S^t

        where the weight is given by

        .. math::

            a^t = 1/t^{\lambda}.

        :param damping:

            Float, see formula above

            Default: :math:`\lambda=.5`


        The ``damping`` :math:`\lambda` is neccessary to assure
        convergence and should be in [0,1]. A default value of 0.5 was
        found to work well in practice. For details, see [HST01]_.

        2. **Rescale the covariance matrix**: Remember that the goal
        is to force the acceptance rate into a specific interval.
        Suppose that the chain already is in a region of significant
        probability mass (should be the case before adapting it).
        When the acceptance rate is close to zero, the chain cannot
        move at all; i.e., the proposed points have a low probability
        relative to the current point. In this case the proposal
        covariance should decrease to increase "locality" of the
        chain.  In the opposite case, when the acceptance rate is
        close to one, the chain most probably only explores a small
        volume of the target.  Then enlarging the covariance matrix
        decreases "locality".  In this implementation, the proposal
        covariance matrix is :math:`c \Sigma^t`

        :param covar_scale_factor:

            Float, this number ``c`` is multiplied to :math:`\Sigma^t`
            after it has been recalculated. The higher the dimension
            :math:`d`, the smaller it should be. For a Gaussian
            proposal and target, the optimal factor is
            :math:`2.38^2/d`. Use this argument to increase
            performance from the start before any adaptation.

            Default: :math:`c=2.38^2/d`


        ``covar_scale_factor`` is updated using :math:`\beta`

        :param covar_scale_multiplier:

            Float;
            if the acceptance rate is larger than ``force_acceptance_max``,
            :math:`c \to \beta c`.
            If the acceptance rate is smaller than ``force_acceptance_min``,
            :math:`c \to c / \beta`.

            Default :math:`\beta=1.5`


        Additionally, an upper and a lower limit on
        ``covar_scale_factor`` can be provided. This is useful to hint
        at bugs in the target or MC implementation that cause the
        efficiency to run away.

        :param covar_scale_factor_max:

            Float, ``covar_scale_factor`` is kept below this value.

            Default: :math:`c_{max}=100`


        :param covar_scale_factor_min:

            Float, ``covar_scale_factor`` is kept above this value.

            Default: :math:`c_{max}=10^{-4}`


        """

        if args != (): raise TypeError('keyword args only; try set_adapt_parameters(keyword = value)')

        self.covar_scale_multiplier = kwargs.pop('covar_scale_multiplier' , self.covar_scale_multiplier)

        self.covar_scale_factor     = kwargs.pop('covar_scale_factor'     , self.covar_scale_factor    )
        self.covar_scale_factor_max = kwargs.pop('covar_scale_factor_max' , self.covar_scale_factor_max)
        self.covar_scale_factor_min = kwargs.pop('covar_scale_factor_min' , self.covar_scale_factor_min)

        self.force_acceptance_max   = kwargs.pop('force_acceptance_max'   , self.force_acceptance_max  )
        self.force_acceptance_min   = kwargs.pop('force_acceptance_min'   , self.force_acceptance_min  )

        self.damping                = kwargs.pop('damping'                , self.damping               )


        if not kwargs == {}: raise TypeError('unexpected keyword(s): ' + str(kwargs.keys()))


    def adapt(self):
        r"""Update the proposal using the points
        stored in ``self.samples[-1]`` and the parameters which can be set via
        :py:meth:`.set_adapt_params`.
        In the above referenced function's docstring, the algorithm is
        described in detail. If the resulting matrix is not a valid covariance,
        its offdiagonal elements are set to zero and a warning is printed. If
        that also fails, the proposal's covariance matrix is divided by the
        ``covar_scale_multiplier`` :math:`\beta`.

        .. note::
            This function only uses the points obtained during the last run.

        """
        last_run = self.samples[-1]
        accept_rate = float(self._last_accept_count) / len(last_run)

        # careful with rowvar!
        # in this form it is expected that each column  of ``points``
        # represents sampling values of a variable
        # this is the case if points is a list of sampled points
        covar_estimator = _np.cov(last_run, rowvar=0)

        # update sigma
        time_dependent_damping_factor = 1./self.adapt_count**self.damping
        self.unscaled_sigma = (1-time_dependent_damping_factor) * self.unscaled_sigma\
                               + time_dependent_damping_factor  * covar_estimator
        self._update_scale_factor(accept_rate)
        scaled_sigma = self.covar_scale_factor * self.unscaled_sigma

        # increase count now before proposal update. It may fail and raise an exception.
        self.adapt_count += 1

        try:
            self.proposal.update(scaled_sigma)
        except _np.linalg.LinAlgError:
            print("WARNING: Markov chain self adaptation failed; trying diagonalization ... ", end='')
            # try to insert offdiagonal elements only
            diagonal_matrix = _np.zeros_like(scaled_sigma)
            _np.fill_diagonal(diagonal_matrix, _np.diag(scaled_sigma))
            try:
                self.proposal.update(diagonal_matrix)
                print('success')
            except _np.linalg.LinAlgError:
                print('fail')
                # just scale the old covariance matrix if everything else fails
                self.proposal.update(self.proposal.sigma / self.covar_scale_multiplier)

    def _update_scale_factor(self, accept_rate):
        '''Private function.
        Updates the covariance scaling factor ``covar_scale_factor``
        according to its limits

        '''
        if accept_rate > self.force_acceptance_max and self.covar_scale_factor < self.covar_scale_factor_max:
            self.covar_scale_factor *= self.covar_scale_multiplier
        elif accept_rate < self.force_acceptance_min and self.covar_scale_factor > self.covar_scale_factor_min:
            self.covar_scale_factor /= self.covar_scale_multiplier
