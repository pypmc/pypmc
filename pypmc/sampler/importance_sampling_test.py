'''Unit tests for the importance sampler functions.

'''

from .importance_sampling import *
from .. import density
from ..density.mixture_test import DummyComponent
from ..tools._probability_densities import unnormalized_log_pdf_gauss, normalized_pdf_gauss
from nose.plugins.attrib import attr
import numpy as np
import unittest
from math import log

rng_seed   = 215135183 # do not change!
rng_steps  = 10000
less_steps = 5

dim = 2
mu  = np.array( [1.     , -1.   ] )
cov = np.array([[11.5   ,  2.875],
                [2.875  ,  0.75 ]])
inv_cov = np.linalg.inv(cov)

log_target = lambda x: unnormalized_log_pdf_gauss(x, mu, inv_cov)

target_samples = np.array([[-5.44709992, -2.75569806],
                           [ 2.71150098, -0.40966545],
                           [ 6.70663397,  0.44663665],
                           [ 3.43498611, -0.4713257 ],
                           [ 5.18926004, -0.39922083],
                           [ 2.19327499, -0.89152452],
                           [ 2.00140479, -0.75817199],
                           [-3.48395737, -2.29824693],
                           [-4.53006187, -2.26075246],
                           [ 8.2430438 ,  0.74320662]])

class FixProposal(density.mixture.MixtureDensity):
    def __init__(self, *args, **kwargs):
        super(FixProposal, self).__init__(*args, **kwargs)
        self.i = 0

    def propose(self, N=1, rng=None):
        i = self.i
        self.i += N
        return target_samples[i:self.i]

perturbed_prop = FixProposal((density.gauss.Gauss(mu+.1, cov+.1),))
perfect_prop   = FixProposal((density.gauss.Gauss(mu, cov),))

def raise_not_implemented(x):
    if (x == np.ones(5)).all():
        return 1.
    raise NotImplementedError()

def unimodal_sampling(instance, ImportanceSamplerClass):
    # test weighted sampling from an unnormalized Gauss using a Student-t proposal

    minus_log_ten_delta_mean = 2
    minus_log_ten_delta_cov  = 3

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
    prop       = density.student_t.StudentT(prop_mean, prop_sigma, prop_dof)

    sam = ImportanceSamplerClass(log_target, prop, prealloc = rng_steps, indicator = None, rng = np.random.mtrand)
    for i in range(10):
        sam.run(rng_steps//10)
    weighted_points = sam.history[:]
    num_samples = len(weighted_points)

    sampled_mean = calculate_mean      (weighted_points)
    sampled_cov  = calculate_covariance(weighted_points)

    instance.assertEqual(num_samples, rng_steps)
    np.testing.assert_almost_equal(sampled_mean, mean , decimal = minus_log_ten_delta_mean)
    np.testing.assert_almost_equal(sampled_cov , sigma, decimal = minus_log_ten_delta_cov )

    # test correcness of weights
    sampled_weights = weighted_points[:,0 ]
    samples         = weighted_points[:,1:]

def bimodal_sampling(instance, ImportanceSamplerClass):
    # test weighted sampling from a bimodal Gaussian mixture using a Student-t mixture proposal
    delta_abun   = .02
    delta_mean   = .02
    delta_sigma1 = .0005
    delta_sigma2 = .02

    # target
    target_abundances = np.array((.6, .4))

    mean1  = np.array( [-5.  , 0.    ])
    sigma1 = np.array([[0.01 , 0.003 ],
                       [0.003, 0.0025]])
    inv_sigma1 = np.linalg.inv(sigma1)

    mean2  = np.array( [+5. , 0.   ])
    sigma2 = np.array([[0.1 , 0.0  ],
                       [0.0 , 0.5  ]])
    inv_sigma2 = np.linalg.inv(sigma2)

    log_target = lambda x: log( target_abundances[0] * normalized_pdf_gauss(x, mean1, inv_sigma1) +
                                target_abundances[1] * normalized_pdf_gauss(x, mean2, inv_sigma2) ) + \
                                15. # break normalization

    # proposal
    prop_abundances = np.array((.5, .5))

    prop_dof1   = 5.
    prop_mean1  = np.array( [-4.9  , 0.01  ])
    prop_sigma1 = np.array([[ 0.007, 0.0   ],
                            [ 0.0  , 0.0023]])
    prop1       = density.student_t.StudentT(prop_mean1, prop_sigma1, prop_dof1)

    prop_dof2   = 5.
    prop_mean2  = np.array( [+5.08, 0.01])
    prop_sigma2 = np.array([[ 0.14, 0.01],
                            [ 0.01, 0.6 ]])
    prop2       = density.student_t.StudentT(prop_mean2, prop_sigma2, prop_dof2)

    prop = density.mixture.MixtureDensity((prop1, prop2), prop_abundances)


    sam = ImportanceSamplerClass(log_target, prop, prealloc = rng_steps, rng = np.random.mtrand)
    for i in range(5):
        sam.run(rng_steps//5)

    weighted_points = sam.history[:]
    num_samples = len(weighted_points)
    instance.assertEqual(num_samples, rng_steps)


    # separate samples from component (+5,0) and (-5,0) by sign of first coordinate
    negative_samples = weighted_points[np.where(weighted_points[:,1]<0.)]
    positive_samples = weighted_points[np.where(weighted_points[:,1]>0.)]

    instance.assertEqual(len(positive_samples) + len(negative_samples), rng_steps)

    # check abundances
    negative_weightsum = negative_samples[:,0].sum()
    positive_weightsum = positive_samples[:,0].sum()
    total_weightsum    = positive_weightsum + negative_weightsum

    instance.assertAlmostEqual(negative_weightsum / total_weightsum, target_abundances[0], delta = delta_abun)
    instance.assertAlmostEqual(positive_weightsum / total_weightsum, target_abundances[1], delta = delta_abun)

    # check means
    sampled_mean1 = calculate_mean(negative_samples)
    sampled_mean2 = calculate_mean(positive_samples)

    np.testing.assert_allclose(sampled_mean1, mean1, atol = delta_mean)
    np.testing.assert_allclose(sampled_mean2, mean2, atol = delta_mean)

    # check covars
    sampled_cov1 = calculate_covariance(negative_samples)
    sampled_cov2 = calculate_covariance(positive_samples)

    np.testing.assert_allclose(sampled_cov1, sigma1, atol = delta_sigma1)
    np.testing.assert_allclose(sampled_cov2, sigma2, atol = delta_sigma2)

#-----------------------------------------------------------------------------------------------------------------

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

    def test_tracing(self):
        dummy_target = lambda x: 0.
        components = []
        for i in range(5):
            components.append( DummyComponent(propose=[float(i)]) )
        prop = density.mixture.MixtureDensity(components)
        sam  = ImportanceSampler(dummy_target, prop)

        origins = sam.run(50, trace_sort=True)
        samples = sam.history[-1]

        for i in range(50):
            self.assertAlmostEqual(samples[i][1], origins[i], delta=1.e-15)

    def test_indicator(self):
        prop = density.gauss.Gauss(np.ones(5), np.eye(5))

        no_ind = ImportanceSampler(raise_not_implemented, prop)
        self.assertRaises(NotImplementedError, no_ind.run)

        indicator = lambda x: (x == np.ones(5)).all()
        with_ind  = ImportanceSampler(raise_not_implemented, prop, indicator)
        with_ind.run()

        weighted_samples = with_ind.history[:]
        weights = weighted_samples[:,0]

        # samples out of support should have zero weight
        np.testing.assert_allclose(weights, 0.)

    @attr('slow')
    def test_unimodal_sampling(self):
        unimodal_sampling(self, ImportanceSampler)

    @attr('slow')
    def test_bimodal_sampling(self):
        bimodal_sampling(self, ImportanceSampler)

    def test_weights(self):
        log_target = lambda x: unnormalized_log_pdf_gauss(x, mu, inv_cov)

        target_weights = np.array([5.64485430502, 4.21621342833, 6.19074100415, 6.57693562598, 1.39850240669])

        sam = ImportanceSampler(log_target, perturbed_prop, rng=np.random.mtrand)
        sam.run(less_steps)
        weights_sampels = sam.history[:]

        weights = weights_sampels[:,0 ]
        samples = weights_sampels[:,1:]

        # samples must be the target_samples --> calculate target_weights by hand
        np.testing.assert_allclose(samples, target_samples[:less_steps])

        np.testing.assert_allclose(weights, target_weights)

class TestDeterministicIS(unittest.TestCase):
    def setUp(self):
        np.random.mtrand.seed(rng_seed)

    @attr('slow')
    def test_unimodal_sampling(self):
        unimodal_sampling(self, DeterministicIS)

    @attr('slow')
    def test_bimodal_sampling(self):
        bimodal_sampling(self, DeterministicIS)

    def test_weights(self):
        target_weights_first  = np.array([5.64485430502, 4.21621342833, 6.19074100415, 6.57693562598, 1.39850240669])

        target_weights_second = np.array([ 4.51833133,  3.97876579,  4.68361755,  4.79001426,  2.03969365,
                                           4.42676502,  4.7814771 ,  4.38248357,  4.42923761,  4.80564581  ])

        sam = DeterministicIS(log_target, perturbed_prop, rng=np.random.mtrand)

        sam.run(less_steps)
        samples_weights_first_step = sam.history[:].copy() # need a copy because weights will be overwritten
        sam.proposal.components[0].update(mu, cov) # set proposal = normalized target (i.e. perfect_prop)
        sam.run(less_steps)
        samples_weights_second_step = sam.history[:]

        # first column is weight -> cut it to get samples only
        samples_first  = samples_weights_first_step [:,1:]
        samples_second = samples_weights_second_step[:,1:]

        weights_first  = samples_weights_first_step [:,0]
        weights_second = samples_weights_second_step[:,0]

        for j in range(dim):
            # first samples should be unchanged (bitwise equal), only the weights should differ
            for i in range(less_steps):
                self.assertEqual(samples_first[i,j] , samples_second[i,j])
            # samples should be the target_samples --> need exactly these samples to calculate by hand
            for i in range(2*less_steps):
                self.assertAlmostEqual(samples_second[i,j], target_samples[i,j])

        # check weights before adaptation
        for i, sample in enumerate(samples_first):
            self.assertAlmostEqual(weights_first[i], target_weights_first[i], places=6)

        # check weights after adaptation
        for i, sample in enumerate(samples_second):
            self.assertAlmostEqual(weights_second[i], target_weights_second[i], places=6)

        # standard weights should not have been saved
        with self.assertRaises(AttributeError):
            sam.std_weights

    def test_clear(self):
        N = 20
        prop = density.mixture.MixtureDensity((density.gauss.Gauss(mu, cov),))
        pmc = DeterministicIS(log_target, prop, rng=np.random.mtrand)
        pmc.run(N)
        pmc.history.clear()
        self.assertRaisesRegexp(AssertionError, r'^Inconsistent state(.*)try ["\'`]*self.clear', pmc.run)
        pmc.clear()
        pmc.run(N)
        weighted_samples = pmc.history[:]
        self.assertEqual(len(weighted_samples), 20)

    def test_std_weights(self):
        target_dmx_weights = np.array([ 4.51833133,  3.97876579,  4.68361755,  4.79001426,  2.03969365,
                                        4.42676502,  4.7814771 ,  4.38248357,  4.42923761,  4.80564581  ])

        target_std_weights = np.array([5.64485430502, 4.21621342833, 6.19074100415, 6.57693562598, 1.39850240669] +
                                      [3.76663727037] * 5)

        sam = DeterministicIS(log_target, perturbed_prop, rng=np.random.mtrand, std_weights=True)
        sam.run(less_steps)
        sam.proposal.components[0].update(mu, cov) # set proposal = normalized target (i.e. perfect_prop)
        sam.run(less_steps)

        # check weights
        dmx_weights = sam.history    [:][:,0]
        std_weights = sam.std_weights[:][:,0]

        for i in range(10):
            self.assertAlmostEqual(dmx_weights[i], target_dmx_weights[i], places=6)
            self.assertAlmostEqual(std_weights[i], target_std_weights[i], places=6)
