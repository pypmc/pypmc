'''Unit tests for the importance sampler functions.

'''

from .importance_sampling import *
from . import proposal
from .._tools._probability_densities import unnormalized_log_pdf_gauss, normalized_pdf_gauss
import numpy as np
import unittest
from math import exp, log

rng_seed  = 215135183
rng_steps = 50000

def raise_not_implemented(x):
    if (x == np.ones(5)).all():
        return 1.
    raise NotImplementedError()

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
    def setUp(self):
        np.random.mtrand.seed(rng_seed)

    def test_indicator(self):
        prop = proposal.GaussianComponent(np.ones(5), np.eye(5))

        self.assertRaises(NotImplementedError, lambda: ImportanceSampler(raise_not_implemented, prop))

        indicator = lambda x: (x == np.ones(5)).all()
        with_ind  = ImportanceSampler(raise_not_implemented, prop, indicator)
        with_ind.run()

        weighted_samples = with_ind.hist[:][1]
        weights = weighted_samples[:,0]

        # samples out of support should have zero weight
        np.testing.assert_allclose(weights, 0.)

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

        sam = ImportanceSampler(log_target, prop, prealloc = rng_steps, indicator = None, rng = np.random.mtrand)
        sam.run(rng_steps)
        num_samples, weighted_points = sam.hist[:]

        sampled_mean = calculate_mean      (weighted_points)
        sampled_cov  = calculate_covariance(weighted_points)

        self.assertEqual(num_samples, rng_steps + 1)
        np.testing.assert_almost_equal(sampled_mean, mean, decimal = minus_log_ten_delta_mean)
        np.testing.assert_almost_equal(sampled_cov , sigma , decimal = minus_log_ten_delta_cov)

        # test correcness of weights
        sampled_weights = weighted_points[:,0]
        samples         = weighted_points[:,1:]

        for i, point in enumerate(samples):
            self.assertAlmostEqual(sampled_weights[i], exp(log_target(point))/exp(prop.evaluate(point)), delta = 1.e-15) #double accuracy

    def test_bimodal_sampling(self):
        # test weighted sampling from a bimodal Gaussian mixture using a Student-t mixture proposal
        delta_abun = .005
        minus_log_ten_delta_mean = 2
        delta_sigma1 = .0005
        delta_sigma2 = .005

        # target
        target_abundancies = np.array((.6, .4))

        mean1  = np.array( [-5.,  0.    ])
        sigma1 = np.array([[0.01 , 0.003 ],
                          [0.003 , 0.0025]])
        inv_sigma1 = np.linalg.inv(sigma1)

        mean2  = np.array( [+5. , 0.   ])
        sigma2 = np.array([[0.1 , 0.0  ],
                           [0.0 , 0.5  ]])
        inv_sigma2 = np.linalg.inv(sigma2)

        log_target = lambda x: log( target_abundancies[0] * normalized_pdf_gauss(x, mean1, inv_sigma1) +
                                    target_abundancies[1] * normalized_pdf_gauss(x, mean2, inv_sigma2) ) + \
                                    15. # break normalization

        # proposal
        prop_abundancies = np.array((.5, .5))

        prop_dof1   = 5.
        prop_mean1  = np.array( [-4.9  , 0.01  ])
        prop_sigma1 = np.array([[ 0.007, 0.0   ],
                                [ 0.0  , 0.0023]])
        prop1       = proposal.StudentTComponent(prop_mean1, prop_sigma1, prop_dof1)

        prop_dof2   = 5.
        prop_mean2  = np.array( [+5.08, 0.01])
        prop_sigma2 = np.array([[ 0.14, 0.01],
                                [ 0.01, 0.6 ]])
        prop2       = proposal.StudentTComponent(prop_mean2, prop_sigma2, prop_dof2)

        prop = proposal.MixtureProposal((prop1, prop2), prop_abundancies)


        sam = ImportanceSampler(log_target, prop, prealloc = rng_steps, rng = np.random.mtrand)
        sam.run(rng_steps)

        num_samples, weighted_points = sam.hist[:]
        self.assertEqual(num_samples, rng_steps + 1)


        # separate samples from component (+5,0) and (-5,0) by sign of first coordinate
        negative_samples = weighted_points[np.where(weighted_points[:,1]<0.)]
        positive_samples = weighted_points[np.where(weighted_points[:,1]>0.)]

        self.assertEqual(len(positive_samples) + len(negative_samples), rng_steps + 1)

        # check abundancies
        negative_weightsum = negative_samples[:,0].sum()
        positive_weightsum = positive_samples[:,0].sum()
        total_weightsum    = positive_weightsum + negative_weightsum

        self.assertAlmostEqual(negative_weightsum / total_weightsum, target_abundancies[0], delta = delta_abun)
        self.assertAlmostEqual(positive_weightsum / total_weightsum, target_abundancies[1], delta = delta_abun)

        # check means
        sampled_mean1 = calculate_mean(negative_samples)
        sampled_mean2 = calculate_mean(positive_samples)

        np.testing.assert_almost_equal(sampled_mean1, mean1, decimal = minus_log_ten_delta_mean)
        np.testing.assert_almost_equal(sampled_mean2, mean2, decimal = minus_log_ten_delta_mean)

        # check covars
        sampled_cov1 = calculate_covariance(negative_samples)
        sampled_cov2 = calculate_covariance(positive_samples)

        np.testing.assert_allclose(sampled_cov1, sigma1, atol = delta_sigma1)
        np.testing.assert_allclose(sampled_cov2, sigma2, atol = delta_sigma2)
