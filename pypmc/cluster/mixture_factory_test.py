"""Unit tests for the mixture factory

"""
from .mixture_factory import *
#from ..pmc.proposal import MixtureProposal, GaussianComponent #TODO: remove
import numpy as np
import unittest

means = np.array([[ 1.0,  5.4, -3.1],
                  [-3.8,  2.5,  0.4],
                  [ 4.1, -3.3, 19.8],
                  [-9.1, 25.4,  1.0]])

covs  = np.array([[[ 3.7,  0.7, -0.6],
                   [ 0.7,  4.5,  0.5],
                   [-0.6,  0.5,  0.6]],

                  [[ 7.0,  1.2,  0.6],
                   [ 1.2,  1.3,  1.5],
                   [ 0.6,  1.5,  4.1]],

                  [[ 1.3,  0.9, -0.3],
                   [ 0.9,  4.1, -0.2],
                   [-0.3, -0.2,  2.2]],

                  [[ 1.6, -0.3, -0.6],
                   [-0.3,  6.6, -0.5],
                   [-0.6, -0.5,  9.4]]])

weights = np.array([ 2.7,  0.4, 1.6, 4.8])

normalized_weights = weights/weights.sum()

class TestCreateGaussian(unittest.TestCase):
    def test_invalid_input(self):
        self.assertRaisesRegexp(AssertionError, 'number of means.*?not match.*?number of cov',
                                create_gaussian_mixture, means    , covs[:2]   )
        self.assertRaisesRegexp(AssertionError, 'number of means.*?not match.*?number of cov',
                                create_gaussian_mixture, means[:2], covs       )

    def test_create_no_weights(self):
        mix = create_gaussian_mixture(means, covs)

        for i, (component, weight) in enumerate(mix):
            self.assertAlmostEqual(weight, .25)
            np.testing.assert_equal(component.mu   , means[i])
            np.testing.assert_equal(component.sigma, covs [i])

    def test_create_with_weights(self):
        mix = create_gaussian_mixture(means, covs, weights)

        for i, (component, weight) in enumerate(mix):
            self.assertAlmostEqual(weight, normalized_weights[i])
            np.testing.assert_equal(component.mu   , means[i])
            np.testing.assert_equal(component.sigma, covs [i])

class TestRecoverGaussian(unittest.TestCase):
    def setUp(self):
        print('when this test fails, first make sure that "create_gaussian_mixture" works')

    def test_recover(self):
        mix = create_gaussian_mixture(means, covs, weights)
        o_means, o_covs, o_weights = recover_gaussian_mixture(mix)
        for i, (component, weight) in enumerate(mix):
            self.assertAlmostEqual(o_weights[i], weight)
            np.testing.assert_equal(o_means[i], component.mu   )
            np.testing.assert_equal(o_covs [i], component.sigma)
