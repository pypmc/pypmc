"""Unit tests for Variational Bayes.

"""

from .variational import *
from .gaussian_mixture import GaussianMixture

import copy
import numpy as np
import unittest

invertible_matrix = np.array([[2.   , 3.    ],
                              [2.   , 2.   ]])

covariance1       = np.array([[0.01 , 0.003 ],
                              [0.003, 0.0025]])

covariance2       = np.array([[0.1  , 0.    ],
                              [0.   , 0.02  ]])

float64_acc = 1.e-15

delta_abun = .01
delta_mean = .5
delta_cov0 = .0002
delta_cov1 = .001


rng_seed = 625135153

class TestGaussianInference(unittest.TestCase):
    def test_update(self):
        np.random.mtrand.seed(rng_seed)

        # generate test data from two independent gaussians
        target_abundances = .5
        target_mean1 = np.array((+5. , 0.))
        target_mean2 = np.array((-5. , 0.))
        target_cov1  = covariance1
        target_cov2  = covariance2
        data1 = np.random.mtrand.multivariate_normal(target_mean1, target_cov1, size = 10**4)
        data2 = np.random.mtrand.multivariate_normal(target_mean2, target_cov2, size = 10**4)
        test_data = np.vstack((data1,data2))
        np.random.shuffle(test_data)

        # provide hint for means to force convergence to a specific solution
        infer = GaussianInference(test_data, 2, m = np.vstack((target_mean1-2.,target_mean2+2.)) )
        infer.update()
        inferred_abundances, inferred_means, inferred_covars = infer.get_result()

        self.assertAlmostEqual(target_abundances, inferred_abundances[0], delta_abun)
        self.assertAlmostEqual(target_abundances, inferred_abundances[1], delta_abun)

        self.assertTrue( (np.abs(target_mean1 - inferred_means[0])<delta_mean).all()  )
        self.assertTrue( (np.abs(target_mean2 - inferred_means[1])<delta_mean).all()  )

        self.assertTrue( (np.abs(target_cov1 - inferred_covars[0])<delta_cov0).all()  )
        self.assertTrue( (np.abs(target_cov2 - inferred_covars[1])<delta_cov1).all()  )

    def test_set_variational_parameters(self):
        infer = GaussianInference(np.empty((20,20)), 5)

        infer.set_variational_parameters(beta0 = 2., W0 = invertible_matrix)

        # set_variational_parameters shall reset not passed parameters to default
        self.assertEqual   (infer.beta0 , 2.)
        self.assertTrue    ( (np.abs(infer.inv_W0-np.linalg.inv(invertible_matrix))<float64_acc).all() )

    def test_prune(self):
        np.random.mtrand.seed(rng_seed)

        # generate test data from two independent gaussians
        target_abundances = np.array((.1, .9))
        target_mean1 = np.array((+1. , -4.))
        target_mean2 = np.array((-5. , +2.))
        target_cov1  = covariance1
        target_cov2  = covariance2
        data1 = np.random.mtrand.multivariate_normal(target_mean1, target_cov1, size =   10**3)
        data2 = np.random.mtrand.multivariate_normal(target_mean2, target_cov2, size = 9*10**3)
        test_data = np.vstack((data1,data2))
        np.random.shuffle(test_data)

        # provide hint for means to force convergence to a specific solution
        infer = GaussianInference(test_data, 3, m = np.vstack((target_mean1+2.,target_mean2+2.,
                                                        np.zeros_like(target_mean1)              )) )
        infer.update()
        infer.prune()
        infer.update()
        inferred_abundances, inferred_means, inferred_covars = infer.get_result()

        # the additional component should have been pruned out
        self.assertEqual(len(inferred_abundances),2)

        self.assertAlmostEqual(target_abundances[0], inferred_abundances[0], delta_abun)
        self.assertAlmostEqual(target_abundances[1], inferred_abundances[1], delta_abun)

        self.assertTrue( (np.abs(target_mean1 - inferred_means[0])<delta_mean).all()  )
        self.assertTrue( (np.abs(target_mean2 - inferred_means[1])<delta_mean).all()  )

        self.assertTrue( (np.abs(target_cov1 - inferred_covars[0])<delta_cov0).all()  )
        self.assertTrue( (np.abs(target_cov2 - inferred_covars[1])<delta_cov1).all()  )

def create_mixture(means, cov, ncomp):
    '''Create mixture density with different means but common covariance.

    For each vector in ``means``, ``ncomp`` Gaussian components are created
    by drawing new means from a multivariate Gaussian with covariance ``cov``.
    Each component has covariance ``cov``.

    '''
    random_centers = np.random.multivariate_normal(means[0], cov, size=ncomp)
    for mu in means[1:]:
        random_centers = np.vstack((random_centers, np.random.multivariate_normal(mu, cov, size=ncomp)))

    return GaussianMixture([GaussianMixture.Component(mu, cov) for mu in random_centers])

class TestVBMerge(unittest.TestCase):

    def setUp(self):
        np.random.mtrand.seed(rng_seed)

    def test_bimodal(self):
        '''Compress bimodal distribution with four components to two components.'''

        means = (np.array([5, 0]), np.array([-5,0]))
        cov = np.eye(2)
        input_components = create_mixture(means, cov, 2)
        initial_guess = create_mixture(means, cov, 1)
        N = 500
        print('input')
        for c in input_components:
            print(c.mean)

        print('initial guess')
        for c in initial_guess:
            print(c.mean)

        vb = VBMerge(input_components, N=N, initial_guess=initial_guess,
                     alpha0=1e-5, beta0=1e-5, nu=np.zeros(2) + 3)

        # initial guess taken over?
        for i,c in enumerate(initial_guess):
            np.testing.assert_array_equal(vb.m[i], c.mean)
            np.testing.assert_array_equal(vb.W[i],  c.inv)

        # all matrices are unit matrices, compute result by hand
        self.assertAlmostEqual(vb.expectation_det_ln_lambda[0], vb.expectation_det_ln_lambda[1])
        self.assertAlmostEqual(vb.expectation_det_ln_lambda[0], 0.84556867019693416)

        # each output comp. should get half of the virtual samples
        self.assertAlmostEqual(vb.N_comp[0], N / 2)

        # all components have same weight initially
        self.assertAlmostEqual(vb.expectation_ln_pi[0], vb.expectation_ln_pi[1])

        # painful calculation by hand
        self.assertAlmostEqual(vb.log_rho[0,0], -18750628.396350645, delta=1e-5)

        old_bound = vb.likelihood_bound()

        vb.update()

        # must improve on initial guess
        self.assertGreater(vb.likelihood_bound(), old_bound)

        # first/second input mapped to first output
        self.assertAlmostEqual(vb.r[0,0], 1)
        self.assertAlmostEqual(vb.r[1,0], 1)

        # third/fourth input mapped to second output
        self.assertAlmostEqual(vb.r[2,1], 1)
        self.assertAlmostEqual(vb.r[3,1], 1)

        # alpha and beta equal after first update
        np.testing.assert_allclose(vb.alpha, vb.N_comp + 1e-5, rtol=1e-15)
        np.testing.assert_allclose(vb.beta, vb.N_comp + 1e-5, rtol=1e-15)

        # mean simply average of two input components
        average = np.array([[ 4.16920636,  1.22792254],
                            [-3.9225488 ,  0.61861674]])
        np.testing.assert_allclose(vb.x_mean_comp[0], average[0], rtol=1e-7)
        np.testing.assert_allclose(vb.x_mean_comp[1], average[1], rtol=1e-7)

        # compute S_0 + C_0 by hand
        S0 =  np.array([[ 1.02835205, -0.12980275],
                        [-0.12980275,  1.5942694 ]])
        np.testing.assert_allclose(vb.S[0], S0)

        # is output properly generated?
        output = vb.get_result()
        self.assertAlmostEqual(output.w[0], 0.5, 13)
        # best fit is just the average
        for i in range(2):
            np.testing.assert_allclose(output[i].mean, average[i], rtol=1e-7)

        # covariance only roughly determined
        for c in output:
            for i in range(2):
                self.assertGreater(c.cov[i, i], 0.5)
                self.assertLess(   c.cov[i, i], 3)
            self.assertAlmostEqual(c.cov[0, 1], c.cov[1, 0])
            self.assertGreater(c.cov[0, 1], -1)
            self.assertLess(   c.cov[0, 1], +1)

        # converge after one step
        old_bound = vb.likelihood_bound()

        vb.update()
        output2 = vb.get_result()

        # it's a discrete problem that converges exactly,
        # so mean and bound are identical
        np.testing.assert_array_equal(output[0].mean, output2[0].mean)
        np.testing.assert_array_equal(output[1].mean, output2[1].mean)
        self.assertAlmostEqual(output.w[0], 0.5, places=5)
        self.assertEqual(vb.likelihood_bound(), old_bound)

        # expect nothing to die out
        vb.prune()
        self.assertEqual(len(output.comp), len(initial_guess.comp))

    def test_large_prune(self):
        '''Compress large number of similar components into a single component.'''

        means = (np.array([5, 0]),)
        cov = np.eye(2)
        N = 500
        N_input = 300
        N_output_initial = 50
        input_components = create_mixture(means, cov, N_input)
        initial_guess = create_mixture(means, cov, N_output_initial)

        vb = VBMerge(input_components, N=N, initial_guess=initial_guess)
        vb_prune = copy.deepcopy(vb)
        bound = vb.likelihood_bound()

        print('Keep all components...\n')

        for i in range(1, 20):
            old_bound = bound
            vb.update()
            bound = vb.likelihood_bound()
            # bound increases only if no component removed
            self.assertGreaterEqual(bound, old_bound)
            if bound == old_bound:
                converged_exactly = True
                break
            print('%d: bound=%g, ncomp=%d' % (i, bound, vb.components))

        self.assertTrue(converged_exactly)
        self.assertEqual(vb.components, N_output_initial)

        # now the same thing with pruning, should be faster
        # as only one component remains
        # but bound can decrease if component is removed
        print('Pruning...\n')
        old_K = vb_prune.components
        bound = vb_prune.likelihood_bound()

        for i in range(1, 20):
            old_bound = bound
            vb_prune.update()
            bound = vb_prune.likelihood_bound()
            print('%d: bound=%g, ncomp=%d, N_k=%s' % (i, bound, vb_prune.components, vb_prune.N_comp))

            if bound == old_bound:
                converged_exactly = True
                break

            self.assertLessEqual(vb_prune.components, old_K)
            if vb_prune.components == old_K:
                print('Comparing bounds')
                self.assertGreaterEqual(bound, old_bound)

            # save K *before* prune()
            old_K = vb_prune.components
            vb_prune.prune()

        self.assertTrue(converged_exactly)
        self.assertEqual(vb_prune.components, 1)

class TestWishart(unittest.TestCase):
    # comparison done in mathematica
    W = np.array([[1,    0.3],
                  [0.3, 11.2]])
    nu = 8.3

    def test_Wishart_log_B(self):
        W = np.eye(3)
        nu = 6

        log_B = Wishart_log_B(W, nu)
        self.assertAlmostEqual(log_B, np.log(0.00013192862453429398))

        log_B = Wishart_log_B(self.W, self.nu)
        self.assertAlmostEqual(log_B, -19.6714760251454)

    def test_Wishart_H(self):
        # compare with Mathematica

        self.assertAlmostEqual(Wishart_H(self.W, self.nu), 11.4262373965875)

    def test_Wishart_expect_log_lambda(self):
        self.assertAlmostEqual(Wishart_expect_log_lambda(self.W, self.nu), 6.24348627492751)
