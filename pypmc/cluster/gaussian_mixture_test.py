"""Unit tests for the Gaussian mixture.

"""
from __future__ import division
from .gaussian_mixture import *
import numpy as np
import unittest

class TestGaussianMixture(unittest.TestCase):
    mean = np.ones(3)
    cov = np.eye(3)

    def test_component(self):
        c = GaussianMixture.Component(self.mean, self.cov, inv=True)
        self.assertAlmostEqual(c.det, 1, 13)
        self.assertTrue(np.allclose(c.inv, self.cov))

    def test_component_invalid(self):
        with self.assertRaises(AssertionError):
            GaussianMixture.Component(np.ones(len(self.cov) + 1), self.cov)

    def test_mixture(self):
        ncomp = 5
        components = [GaussianMixture.Component(self.mean, self.cov) for i in range(ncomp)]
        # automatic normalization
        mix = GaussianMixture(components)
        self.assertTrue(mix.normalized())
        self.assertAlmostEqual(mix.w[0], 1 / ncomp)

        # blow normalization
        mix.w[0] = 2
        self.assertFalse(mix.normalized())

        # renormalize
        mix.normalize()
        self.assertTrue(mix.normalized())

        # removing elements
        mix.w[0] = 0.0
        mix.w[1] = 0.5
        mix.normalize()
        self.assertEqual(mix.prune(), [0])
        self.assertEqual(len(mix.w), ncomp - 1)
        self.assertTrue(mix.normalized())

