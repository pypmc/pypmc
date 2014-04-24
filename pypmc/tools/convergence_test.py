'Unit tests for convergence diagnostics'

import numpy as np
from .convergence import *

import unittest

weights = np.array([0., 1., 2., 3., 4.])
weights_integer = range(5)

class TestPerplexity(unittest.TestCase):
    target = 0.71922309332486445

    def test_float(self):
        self.assertAlmostEqual(perp(weights), self.target)

    def test_truediv(self):
        self.assertAlmostEqual(perp(weights_integer), self.target)

class TestESS(unittest.TestCase):
    target = 2. / 3.

    def test_float(self):
        self.assertAlmostEqual(ess(weights), self.target)

    def test_truediv(self):
        self.assertAlmostEqual(ess(weights_integer), self.target)
