'''Unit tests for the importance sampler functions.

'''

from .importance_sampling import *
from . import proposal
import numpy as np
import unittest
from math import exp

rng_seed  = 215135183
rng_steps = 50000

def unnormalized_log_pdf_gauss(x, mu, inv_sigma):
    return - .5 * (x-mu).dot(inv_sigma).dot(x-mu)

class TestCalculateExpextaction(unittest.TestCase):
    def test_calculate(self):
        points                       = np.array([[0. , 4.5 ],
                                                 [4. , 5.5 ],
                                                 [2. , 5.  ]])

        weights                      = np.array([[1. ,2. ,5. ]]).transpose()

        weighted_points = np.hstack((weights, points))

        mean1       = calculate_expectation(weighted_points, lambda x: x)
        mean2       = calculate_mean(weighted_points)
        target_mean = np.array([2.25, 5.0625])

        np.testing.assert_almost_equal(mean1, target_mean, decimal = 15) #double accuracy
        np.testing.assert_almost_equal(mean2, target_mean, decimal = 15) #double accuracy

        cov        = calculate_covariance(weighted_points)
        target_cov = 8./34. * np.array([[11.5   , 2.875  ],
                                        [2.875  , 0.71875]])

        np.testing.assert_almost_equal(cov, target_cov, decimal = 15) #double accuracy

class TestImportanceSampler(unittest.TestCase):
    @unittest.skip('must be written first')
    def test_indicator(self):
        pass

    def test_unimodal_sampling(self):
        # test weighted sampling from an unnormalized Gauss using a Student-t proposal

        minus_log_ten_delta_mean = 3
        minus_log_ten_delta_cov  = 4

        # target
        mean  = np.array( [-4.4,  2.8   ])
        sigma = np.array([[0.01 , 0.003 ]
                         ,[0.003, 0.0025]])
        inv_sigma = np.linalg.inv(sigma)
        log_target = lambda x: unnormalized_log_pdf_gauss(x, mean, inv_sigma)

        # proposal
        prop_dof   = 5.
        prop_mean  = np.array([-4.3, 2.9])
        prop_sigma = np.array([[0.007, 0.0   ]
                              ,[0.0  , 0.0023]])
        prop       = proposal.StudentTComponent(prop_mean, prop_sigma, prop_dof)

        np.random.seed(rng_seed)

        sam = ImportanceSampler(log_target, prop, prealloc = rng_steps, indicator = None, rng = np.random)
        sam.run(rng_steps)
        accept_count, weighted_points = sam.hist[:]

        sampled_mean = calculate_mean      (weighted_points)
        sampled_cov  = calculate_covariance(weighted_points)

        self.assertEqual(accept_count, rng_steps + 1)
        np.testing.assert_almost_equal(sampled_mean, mean, decimal = minus_log_ten_delta_mean)
        np.testing.assert_almost_equal(sampled_cov , sigma , decimal = minus_log_ten_delta_cov)

        # test correcness of weights
        sampled_weights = weighted_points[:,0]
        samples         = weighted_points[:,1:]

        for i, point in enumerate(samples):
            self.assertAlmostEqual(sampled_weights[i], exp(log_target(point))/exp(prop.evaluate(point)), delta = 1.e-15) #double accuracy

    @unittest.skip('must be written first')
    def test_bimodal_sampling(self):
        pass









