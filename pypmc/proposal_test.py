"""Unit tests for the MCMC proposal functions.

"""

from proposal import *
import numpy as np
import unittest

class TestGaussian(unittest.TestCase):
    def test_evaluate(self):
        sigma = np.matrix([[0.01, 0.003],[0.003, 0.0025]])
        t = MultivariateStudentT(mu=np.zeros(2), sigma=sigma)

        x = np.array([4.3, 1.1])
        y = np.array([4.35, 1.2])

        target = 1.30077135
        self.assertAlmostEqual(t.evaluate(x, y), 0, delta=1e-8)
        self.assertAlmostEqual(t.evaluate(y, x), 0, delta=1e-8)

class TestStudentT(unittest.TestCase):
    pass
