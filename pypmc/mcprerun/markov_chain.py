"""Collect Markov Chain"""

import numpy as _np
from .._tools import _inherit_docstring

class _Chain(object):
    """Abstract base class implementing a sequence of points

    """
    def run(self, N = 1):
        '''Runs the chain and stores the history of visited points into
        the member variable self.points

        :param N:

            An int which defines the number of steps to run the chain.

        '''
        raise NotImplementedError()

    def adapt(self):
        """Update the proposal's covariance matrix using the points
        stored in self.points

        """
        return self.proposal.adapt(self.points)


    def clear(self):
        """Delete all history of visited points except the last one

        """
        self.points = [self.points[-1]]

class MarkovChain(_Chain):
    """MarkovChain(target, proposal, start, indicator = None,
    rng = numpy.random.mtrand)\n
    A Markov chain with adaptive proposal density

    :param target:

        The target density. Must be a function which recieves a 1d numpy
        array and returns a float, namely log(P(x)) the log of the target.

    :param proposal:

        The proposal density.
        Should be of type :py:class:`pypmc.mcprerun.proposal.ProposalDensity`.

        .. hint::
            When your proposal density is symmetric, define the member
            variable ``proposal.symmetric = True``. This will omit calls
            to proposal.evaluate

    :param start:

        The starting point of the Markov chain. (numpy array)

    :param indicator:

        A function wich recieves a numpy array and returns bool.
        The target is only called if indicator(proposed_point)
        returns True, otherwise the proposed point is rejected
        without call to target.
        Use this function to specify the support of the target.

        .. seealso::
            :py:mod:`pypmc.mcprerun.indicator_factory`

    :param rng:

        The rng passed to the proposal when calling proposal.propose

        .. important::
            ``rng`` must return a numpy array of N samples from the
            uniform distribution [0,1) when calling **rng.rand(N)**

        .. seealso::
            ``rng`` must also fulfill the requirements of
            :py:meth:`pypmc.mcprerun.proposal.ProposalDensity.propose`

    """
    def __init__(self, target, proposal, start, indicator = lambda x: True, rng = _np.random.mtrand):
        # store input into instance
        self.target    = target
        self.proposal  = proposal
        self.points    = [start]
        self.indicator = indicator
        self.rng       = rng
        self.symmetric = self.proposal.symmetric

    @_inherit_docstring(_Chain)
    def run(self, N = 1, adapt = 0):

        i_adapt = 0

        for i_N in range(N):
            # adapt proposal
            i_adapt += 1
            if i_adapt == adapt:
                i_adapt = 0
                self.proposal.adapt(self.points)

            # propose new point
            proposed_point = self.proposal.propose(self.points[-1], self.rng)

            # if self.indicator returns False reject the point
            if not self.indicator(proposed_point):
                self.points.append(self.points[-1])
                continue

            log_target_ratio = self.target(proposed_point) - self.target(self.points[-1])

            # log_rho := log(probability to accept point), where log_rho > 0 is meant to imply rho = 1
            if self.proposal.symmetric:
                log_rho = log_target_ratio
            else:
                log_proposal_ratio = self.proposal.evaluate(self.points[-1], proposed_point)\
                                   - self.proposal.evaluate(proposed_point, self.points[-1])
                log_rho = log_target_ratio + log_proposal_ratio

            # check for NaN
            if _np.isnan(log_rho): raise ValueError('encountered NaN')

            if log_rho >=0: #accept if rho = 1
                self.points.append(proposed_point)
            elif log_rho >= _np.log(self.rng.rand()): #accept with probability rho
                self.points.append(proposed_point)
            else: #reject if not accepted
                self.points.append(self.points[-1])
