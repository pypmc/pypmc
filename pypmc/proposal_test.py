"""Unit tests for the MCMC proposal functions.

"""

from proposal import *
import numpy as np
import unittest

class TestGaussian(unittest.TestCase): #numbers as in eos
    def test_evaluate(self):
        sigma = np.array([[0.01, 0.003],[0.003, 0.0025]])
        t = MultivariateGaussian(mu=np.zeros(2), sigma=sigma)

        x = np.array([4.3, 1.1])
        y = np.array([4.35, 1.2])

        target = 1.30077135

        self.assertAlmostEqual(t.evaluate(x, y), target, delta=1e-8)
        self.assertAlmostEqual(t.evaluate(y, x), target, delta=1e-8)

    def test_propose(self):
        sigma = np.array([[0.01, 0.003],[0.003, 0.0025]])
        t = MultivariateGaussian(mu=np.zeros(2), sigma=sigma)
        np.random.seed(121501257)
        N = 50000
        current = np.array([4.3, 1.1])
        target_mean1 = 4.3
        target_mean2 = 1.1
        target_chiSq = 2.0
        normalization = -np.log(2. * np.pi) - 0.5 * np.log(1.6e-5)

        mean1 = 0.
        mean2 = 0.
        chiSq = 0.

        for i in xrange(N):
            proposal = t.propose(current, np.random.rand)
            mean1 += proposal[0]
            mean2 += proposal[1]
            chiSq += -2.0 * (t.evaluate(proposal, current) - normalization)

        mean1 /= N
        mean2 /= N
        chiSq /= N

        self.assertAlmostEqual(mean1, target_mean1, delta=.001)
        self.assertAlmostEqual(mean2, target_mean2, delta=.001)
        self.assertAlmostEqual(chiSq, target_chiSq, delta=.001)

class TestStudentT(unittest.TestCase): #numbers as in eos
    @unittest.skip("not Implemented yet")
    def test_evaluate(self):
        pass
