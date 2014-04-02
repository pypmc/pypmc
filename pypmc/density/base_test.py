"""Unit tests for the base class probability densities"""

from .base import *
import numpy as np
import unittest

# dummy proposal component (convenient for testing):
#   - evaluates to the first item in x
class DummyDensity(ProbabilityDensity):
    def __init__(self):
        pass
    def evaluate(self, x):
        return x[0]

class TestProbabilityDensity(unittest.TestCase):
    def setUp(self):
        self.density = DummyDensity()
        self.evaluate_at = np.array([[0.1, 3.4, 6.8, 5.8],
                                     [1.2, 5.9, 1.5, 4.6],
                                     [7.5, 4.5, 6.9, 7.2]])
        self.target = np.array( (0.1, 1.2, 7.5) )

    def test_multi_evaluate_with_out(self):
        out1 = np.empty(3)
        out2 = self.density.multi_evaluate(self.evaluate_at, out1)
        np.testing.assert_allclose(out1, self.target)
        np.testing.assert_allclose(out2, self.target)
        assert out1 is out2

    def test_multi_evaluate_no_out(self):
        out = self.density.multi_evaluate(self.evaluate_at)
        np.testing.assert_allclose(out, self.target)
