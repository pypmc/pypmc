'''Collect abstract base probability density classes

'''

import numpy as _np

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

            Vector-like array; the proposed point.

        """
        raise NotImplementedError()

    def propose(self, N=1, rng=_np.random.mtrand):
        """propose(self, N=1, rng=numpy.random.mtrand)

        Propose N points using the random number generator ``rng``.

        :param N:

            Integer; the number of random points to be proposed

        :param rng:

            State of a random number generator like numpy.random.mtrand

        """
        raise NotImplementedError()

class LocalDensity(object):
    """Abstract base class for a local probability density. Can be used as
    proposal for the Markov chain sampler.

    """
    dim = 0
    symmetric = False

    def __init__(self):
        raise NotImplementedError('Do not create instances from this class, use derived classes instead.')

    def evaluate(self, x, y):
        """Evaluate log of the density to propose ``x`` given ``y``,
        namely log(q(x|y)).

        :param x:

            Vector-like array; the proposed point

        :param y:

            Vector-like array; the current point

        """
        raise NotImplementedError()

    def propose(self, y, rng=_np.random.mtrand):
        """propose(self, y, rng=numpy.random.mtrand)
        Propose a new point given ``y`` using the random number
        generator ``rng``.

        :param y:

            Vector-like array; the current position

        :param rng:

            State of a random number generator like numpy.random.mtrand

        """
        raise NotImplementedError()
