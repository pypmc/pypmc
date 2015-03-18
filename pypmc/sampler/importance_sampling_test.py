'''Unit tests for the importance sampler functions.

'''

from .importance_sampling import *
from .. import density
from ..density.mixture_test import DummyComponent
from ..tools._probability_densities import unnormalized_log_pdf_gauss, normalized_pdf_gauss
from ..tools import History
from nose.plugins.attrib import attr
import numpy as np
import unittest
from math import exp, log

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
    weights = sam.weights[:][:,0]
    samples = sam.samples[:]

    instance.assertEqual(len(samples), len(weights))
    num_samples = len(samples)

    sampled_mean = calculate_mean      (samples, weights)
    sampled_cov  = calculate_covariance(samples, weights)

    instance.assertEqual(num_samples, rng_steps)
    np.testing.assert_almost_equal(sampled_mean, mean , decimal = minus_log_ten_delta_mean)
    np.testing.assert_almost_equal(sampled_cov , sigma, decimal = minus_log_ten_delta_cov )

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

    weights = sam.weights[:][:,0]
    samples = sam.samples[:]

    instance.assertEqual(len(weights), len(samples))
    num_samples = len(weights)
    instance.assertEqual(num_samples, rng_steps)


    # separate samples from component (+5,0) and (-5,0) by sign of first coordinate
    negative_weights = weights[np.where(samples[:,0]<0.)]
    negative_samples = samples[np.where(samples[:,0]<0.)]
    positive_weights = weights[np.where(samples[:,0]>0.)]
    positive_samples = samples[np.where(samples[:,0]>0.)]

    instance.assertEqual(len(positive_weights) + len(negative_weights), rng_steps)

    # check abundances
    negative_weightsum = negative_weights.sum()
    positive_weightsum = positive_weights.sum()
    total_weightsum    = positive_weightsum + negative_weightsum

    instance.assertAlmostEqual(negative_weightsum / total_weightsum, target_abundances[0], delta = delta_abun)
    instance.assertAlmostEqual(positive_weightsum / total_weightsum, target_abundances[1], delta = delta_abun)

    # check means
    sampled_mean1 = calculate_mean(negative_samples, negative_weights)
    sampled_mean2 = calculate_mean(positive_samples, positive_weights)

    np.testing.assert_allclose(sampled_mean1, mean1, atol = delta_mean)
    np.testing.assert_allclose(sampled_mean2, mean2, atol = delta_mean)

    # check covars
    sampled_cov1 = calculate_covariance(negative_samples, negative_weights)
    sampled_cov2 = calculate_covariance(positive_samples, positive_weights)

    np.testing.assert_allclose(sampled_cov1, sigma1, atol = delta_sigma1)
    np.testing.assert_allclose(sampled_cov2, sigma2, atol = delta_sigma2)

#-----------------------------------------------------------------------------------------------------------------

class TestCalculateExpextaction(unittest.TestCase):
    samples = np.array([[0. , 4.5 ],
                        [4. , 5.5 ],
                        [2. , 5.  ]])

    weights = np.array([1. ,2. ,5. ])

    def test_error_messages(self):
        too_many_weights = [1.,2.,3.,4.]

        with self.assertRaisesRegexp(AssertionError, ".*number of samples.*must.*equal.*number of weights"):
            calculate_expectation(self.samples, too_many_weights, lambda x: x)

        with self.assertRaisesRegexp(AssertionError, ".*number of samples.*must.*equal.*number of weights"):
            calculate_mean(self.samples, too_many_weights)

        with self.assertRaisesRegexp(AssertionError, ".*number of samples.*must.*equal.*number of weights"):
            calculate_covariance(self.samples, too_many_weights)

    def test_calculate(self):
        mean1       = calculate_expectation(self.samples, self.weights, lambda x: x)
        mean2       = calculate_mean(self.samples, self.weights)
        target_mean = np.array([2.25, 5.0625])

        np.testing.assert_almost_equal(mean1, target_mean, decimal = 15) #double accuracy
        np.testing.assert_almost_equal(mean2, target_mean, decimal = 15) #double accuracy

        cov        = calculate_covariance(self.samples, self.weights)
        target_cov = 8./34. * np.array([[11.5   , 2.875  ],
                                        [2.875  , 0.71875]])

        np.testing.assert_almost_equal(cov, target_cov, decimal = 15) #double accuracy

def check_save_target_values(test_case, sampler_type):
    N = 10

    prop_mean  = np.array( [4.1 , 1.  ])
    prop_sigma = np.array([[0.1 , 0.  ]
                           ,[0.  , 0.02]])

    prop = density.gauss.Gauss(prop_mean, prop_sigma)

    target_sigma = np.array([[0.1 , 0.04]
                            ,[0.04, 0.02]])
    target_mean  = np.array([4.3, 1.1])
    log_target = lambda x: unnormalized_log_pdf_gauss(x, target_mean, np.linalg.inv(target_sigma))

    sampler = sampler_type(log_target, prop, prealloc=N)
    assert sampler.target_values is None
    sampler = sampler_type(log_target, prop, prealloc=N, save_target_values=True)

    sampler.run(N)
    test_case.assertEqual(len(sampler.samples[-1]), N)
    test_case.assertEqual(len(sampler.weights[-1]), N)

    samples       = sampler.samples[:]
    target_values = sampler.target_values[:]

    test_case.assertEqual(len(sampler.samples), 1)
    test_case.assertEqual(len(sampler.target_values), 1)

    test_case.assertEqual(len(sampler.samples[:]), N)
    test_case.assertEqual(len(sampler.target_values[:]), N)

    for i in range(10):
        # check if target values are correctly saved
        test_case.assertEqual(log_target(samples[i]), sampler.target_values[:][i,0])

        # check if weights are calculated correctly
        test_case.assertAlmostEqual(exp(log_target(samples[i])) / exp(prop.evaluate(samples[i])), sampler.weights[:][i,0], delta=1e-15)

    sampler.clear()

    test_case.assertEqual(len(sampler.target_values), 0)
    test_case.assertEqual(len(sampler.samples), 0)
    test_case.assertEqual(len(sampler.weights), 0)

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
        samples = sam.samples[-1]

        for i in range(50):
            self.assertAlmostEqual(samples[i][0], origins[i], delta=1.e-15)

    def test_indicator(self):
        prop = density.gauss.Gauss(np.ones(5), np.eye(5))

        no_ind = ImportanceSampler(raise_not_implemented, prop)
        self.assertRaises(NotImplementedError, no_ind.run)

        indicator = lambda x: (x == np.ones(5)).all()
        with_ind  = ImportanceSampler(raise_not_implemented, prop, indicator)
        with_ind.run()

        weights = with_ind.weights[:][:,0]

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

        weights = sam.weights[:][:,0 ]
        samples = sam.samples[:]

        # samples must be the target_samples --> calculate target_weights by hand
        np.testing.assert_allclose(samples, target_samples[:less_steps])

        np.testing.assert_allclose(weights, target_weights)

    def test_save_target_values(self):
        check_save_target_values(self, ImportanceSampler)

class TestCombineWeights(unittest.TestCase):
    # one dim samples
    # negative weights check _combine_weights_linear
    #                                 weights     positions
    weighted_samples_1 = np.array([[-5.44709992, -2.75569806],
                                   [ 2.71150098, -0.40966545],
                                   [ 6.70663397,  0.44663665],
                                   [ 3.43498611, -0.4713257 ],
                                   [ 5.18926004, -0.39922083],
                                   [ 2.19327499, -0.89152452]])
    weighted_samples_2 = np.array([[ 2.00140479, -0.75817199],
                                   [-3.48395737, -2.29824693],
                                   [-4.53006187, -2.26075246],
                                   [ 8.2430438 ,  0.74320662]])

    def setUp(self):
        np.random.mtrand.seed(rng_seed)

    def test_combine_weights_linear(self):
        prop1 = density.gauss.Gauss([0.0], [1.0]) # standard Gauss
        prop2 = density.student_t.StudentT(mu=[0.0], sigma=[1.0], dof=1.0)

        target_combined_weights = np.array([
                                  -2.41555928,  3.02214411,  7.5019662 ,  3.85165843,  5.77794706, 2.53678258,
                                   1.55337057, -4.72862573, -5.98542884,  6.41159914
                                  ])

        combined_weights = combine_weights([self.weighted_samples_1[:,1:], self.weighted_samples_2[:,1:]],
                                           [self.weighted_samples_1[:,0], self.weighted_samples_2[:,0]],
                                            [prop1, prop2])
        assert type(combined_weights) is History
        self.assertEqual(combined_weights[:].shape, (10,1))
        for i in range(6):
            self.assertAlmostEqual(combined_weights[0][i,0], target_combined_weights[i])
        for i in range(4):
            self.assertAlmostEqual(combined_weights[1][i,0], target_combined_weights[6 + i])

    def test_combine_weights_log(self):
        target_combined_weights = np.array([ 4.51833133,  3.97876579,  4.68361755,  4.79001426,  2.03969365,
                                             4.42676502,  4.7814771 ,  4.38248357,  4.42923761,  4.80564581  ])

        first_proposal = perturbed_prop
        second_proposal = perfect_prop

        sam = ImportanceSampler(log_target, first_proposal)

        sam.run(less_steps)
        sam.proposal.components[0].update(mu, cov) # set proposal = normalized target (i.e. perfect_prop)
        sam.run(less_steps)

        weights_1 = sam.weights[0][:,0]
        weights_2 = sam.weights[1][:,0]
        samples_1 = sam.samples[0]
        samples_2 = sam.samples[1]

        samples_combined = np.vstack([samples_1, samples_2])

        # positive weights => check _combine_weights_log
        combined_weights = combine_weights([samples_1, samples_2],
                                           [weights_1, weights_2],
                                           #[weighted_samples_1, weighted_samples_2],
                                           [first_proposal    , second_proposal   ])

        for j in range(dim):
            # samples should be the target_samples --> need exactly these samples to calculate by hand
            for i in range(2*less_steps):
                self.assertAlmostEqual(samples_combined[i,j], target_samples[i,j])

        for i, target_weight_i in enumerate(target_combined_weights):
            self.assertAlmostEqual(combined_weights[:][i,0], target_weight_i, places=6)

    def test_error_messages(self):
        # add a zero column to the end
        weighted_samples_1 = np.hstack([self.weighted_samples_1, np.zeros((len(self.weighted_samples_1), 1))])
        weighted_samples_2 = np.hstack([self.weighted_samples_2, np.zeros((len(self.weighted_samples_2), 1))])


        # should be OK
        combine_weights([weighted_samples_1[:,1:], weighted_samples_2[:,1:]],
                        [weighted_samples_1[:,0],  weighted_samples_2[:,0]],
                        [perfect_prop, perfect_prop])

        with self.assertRaisesRegexp(AssertionError, 'Got 2 importance-sampling runs but 1 proposal densities'):
            combine_weights([weighted_samples_1[:,1:], weighted_samples_2[:,1:]],
                            [weighted_samples_1[:,0],  weighted_samples_2[:,0]],
                            [perfect_prop])

        with self.assertRaisesRegexp(AssertionError, 'Got 2 importance-sampling runs but 1 weights'):
            combine_weights([weighted_samples_1[:,1:], weighted_samples_2[:,1:]],
                            [weighted_samples_1[:,0]],
                            [perfect_prop, perfect_prop])

        with self.assertRaisesRegexp(AssertionError, "``samples\[0\]`` is not matrix like."):
            combine_weights([range(9), weighted_samples_2[:,1:]],
                            [weighted_samples_1[:,0],  weighted_samples_2[:,0]],
                            [perfect_prop, perfect_prop])

        with self.assertRaisesRegexp(AssertionError, "Dimension of samples\[0\] \(2\) does not match the dimension of samples\[1\] \(1\)"):
            combine_weights([weighted_samples_1[:,1:], weighted_samples_2[:,1:2]],
                            [weighted_samples_1[:,0],  weighted_samples_2[:,0]],[perfect_prop, perfect_prop])
