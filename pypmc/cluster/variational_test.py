"""Unit tests for Variational Bayes.

"""

from .variational import *
from .gaussian_mixture import GaussianMixture
from .. import pmc
from ..tools._probability_densities import unnormalized_log_pdf_gauss, normalized_pdf_gauss

import copy
from nose.plugins.attrib import attr
import numpy as np
from scipy.special import digamma
import unittest

def check_bound(test_case, variational, n=20, prune=True):
    bound = variational.likelihood_bound()
    old_K = variational.K
    for i in range(n):
        old_bound = bound
        variational.update()
        bound = variational.likelihood_bound()
        print('%d: bound=%.16f, ncomp=%d, N_k=%s' % (i, bound, variational.K, variational.N_comp))

        if bound == old_bound:
            return i + 1

        test_case.assertLessEqual(variational.K, old_K)
        if variational.K == old_K:
            test_case.assertGreaterEqual(bound, old_bound)

        # save K *before* prune()
        old_K = variational.K
        if prune:
            variational.prune()

    return None

invertible_matrix = np.array([[2.   , 3.   ],
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
    def setUp(self):
        np.random.seed(rng_seed)

    def test_parameter_validation(self):
        target_mean = np.array((+1. , -4.))
        data = np.random.mtrand.multivariate_normal(target_mean, covariance1, size=500)
        D = len(target_mean)
        K = 5
        d = dict

        # correct values should not throw
        kwargs = [d(alpha0=2), d(beta0=1e-3), d(nu0=5),
                  d(alpha0=np.zeros(K) + 1e-6), d(beta0=np.zeros(K) + 1e-6), d(nu0=np.zeros(K) + 3),
                  d(alpha=np.zeros(K) + 1e-6), d(beta=np.zeros(K) + 1e-6), d(nu=np.zeros(K) + 3),
                  d(m0=np.zeros(D)), d(W0=np.eye(D)),
                  d(m0=np.zeros((K,D))), d(W0=np.array([np.eye(D)] * K)),
                  d(m=np.zeros((K,D))), d(W=np.array([np.eye(D)] * K)),
                 ]
        for kw in kwargs:
            print(kw)
            GaussianInference(data, K, **kw)

        # unknown argument
        with self.assertRaises(TypeError) as cm:
            GaussianInference(data, K, Balpha0=0.1)
        print(cm.exception)

        # invalid kwargs
        kwargs = [d(alpha0=-2), d(beta0=-1), d(nu0=-1e-6), # wrong value
                  d(alpha0=np.zeros(K)), d(beta0=np.zeros(K)), d(nu0=np.zeros(K)), # wrong values
                  d(alpha=np.zeros(K-1)), d(beta=np.zeros(K+1)), d(nu=np.zeros(1)), # wrong dim
                  d(alpha=np.zeros((K, D)) + 1e-6), # wrong shape
                  d(alpha=np.zeros(K) - 1), d(beta=np.zeros(K) - 1), d(nu=np.zeros(K) + 2), # wrong values
                  d(m0=np.zeros(D + 1)), d(W0=np.eye(D + 3)),
                  d(m=np.zeros(D)), d(W=np.zeros((D, D))), # need to specify all components
                  d(m=np.zeros((K, D + 1))), d(W=np.zeros((K, D + 1, D))),
                  d(m0=np.zeros((K + 1, D)))
                 ]
        for kw in kwargs:
            with self.assertRaises(ValueError) as cm:
                GaussianInference(data, K, **kw)
            print(cm.exception)

        # zero matrix not positive definite
        with self.assertRaises(AssertionError) as cm:
            GaussianInference(data, K, W=np.zeros((K, D, D)))
        print(cm.exception)

    def test_update(self):
        demo_data = np.array([
                              # first component at [0,5]
                              [-2.,  3.],
                              [ 2.,  5.],
                              [-1.,  7.],
                              [ 0.,  4.],
                              [ 1.,  6.],
                              # second component at [0,-5]
                              [ 2., -3.],
                              [-1., -6.],
                              [ 1., -4.],
                              [-2., -7.]])

        target_means   = [np.array([0.,5.]), np.array([0., -5.])]
        target_covars  = [np.array([[2.5,0.75],[0.75,2.5]]), 2.5*np.ones((2,2))]
        target_weights = np.array([0.5,0.5])

        # provide hint for means to force convergence to a specific solution
        alpha0, beta0, nu0 = 1e-5, 1e-5, 3
        infer = GaussianInference(demo_data, 2, m = np.vstack((target_means[0]-2.,target_means[1]+2.)),
                                  alpha0=alpha0, beta0=beta0, nu0=nu0)

        # check W0 and inv_W0
        W0 = inv_W0 = np.eye(2)
        np.testing.assert_allclose(infer.W0[0], W0)
        np.testing.assert_allclose(infer.inv_W0[0], inv_W0)
        np.testing.assert_allclose(infer.W[0], W0)
        np.testing.assert_allclose(infer.W[1], W0)

        # check for correctness of first E-step (called from constructor) --> compare with calculation by hand
        # (only for first data_point if for all points)

        # check self._update_expectation_gauss_exponent()
        exp_gauss_expo = 2.*10**5 + 3.*52
        self.assertAlmostEqual(infer.expectation_gauss_exponent[0,1], exp_gauss_expo, delta = float64_acc*exp_gauss_expo)

        exp_gauss_expo = 2.*10**5 + 3.*20
        self.assertAlmostEqual(infer.expectation_gauss_exponent[1,0], exp_gauss_expo, delta = float64_acc*exp_gauss_expo)

        exp_gauss_expo = 2.*10**5 + 0
        self.assertAlmostEqual(infer.expectation_gauss_exponent[0,0], exp_gauss_expo, delta = float64_acc*exp_gauss_expo)


        # check self._update_expectation_det_ln_lambda()
        exp_det_ln_lambda = digamma(3./2.) + digamma(1.) + 2.*np.log(2)
        self.assertAlmostEqual(infer.expectation_det_ln_lambda[0], exp_det_ln_lambda, delta = float64_acc*exp_det_ln_lambda)

        # check self._update_expectation_ln_pi()
        exp_ln_pi = digamma(1e-5) - digamma(2e-5)
        self.assertAlmostEqual(infer.expectation_ln_pi[0], exp_ln_pi, delta = float64_acc*exp_ln_pi)

        # check self._update_r()
        # log_rho
        log_rho = exp_ln_pi + .5*exp_det_ln_lambda - np.log(2*np.pi) - .5*exp_gauss_expo
        self.assertAlmostEqual(infer.log_rho[0,0], log_rho)

        # check r
        r = 1.3336148155022614e-34
        self.assertAlmostEqual(infer.r[0,1], r)
        self.assertAlmostEqual(infer.r[0,0], 1)

        # check N_comp
        N_comp = np.array([5.,4.])
        np.testing.assert_allclose(infer.N_comp, N_comp)

        # check inv_N_comp
        self.assertEqual(infer.inv_N_comp[0], 1./5.)

        # check x_mean_comp
        x_mean_comp = np.einsum('k,nk,ni->ki', 1./N_comp, infer.r, demo_data)
        np.testing.assert_allclose(infer.x_mean_comp, x_mean_comp)

        # check S
        S = np.zeros((2,2))
        for n in range(9):
            tmp = demo_data[n] - x_mean_comp[0]
            S += infer.r[n,0]*np.outer(tmp,tmp)
        S /= 5.
        np.testing.assert_allclose(infer.S[0], S)


        # check M-step
        infer.M_step()

        nu = nu0 + N_comp
        np.testing.assert_allclose(infer.nu, nu)

        # check beta
        beta = N_comp + beta0
        np.testing.assert_allclose(infer.beta, beta)

        # check m
        m = np.einsum('k,k,ki->ki',1./beta, N_comp, x_mean_comp)
        np.testing.assert_allclose(infer.m, m)

        # check W
        inv_W = inv_W0 + N_comp[0]*S + (infer.beta0*N_comp[0]) / (infer.beta0 + N_comp[0]) *\
                np.outer(x_mean_comp[0], x_mean_comp[0])
        W     = np.linalg.inv(inv_W)
        np.testing.assert_allclose(infer.W[0], W)

    def test_weighted(self):
        # this test uses pypmc.pmc.importance_sampling --> before debugging here,
        # first make sure that importance_sampling works

        # -------------------------------- generate weighted test data ----------------------------------
        # target
        target_abundancies = np.array((.7, .3))

        mean1  = np.array( [-5.   , 0.    ])
        sigma1 = np.array([[ 0.01 , 0.003 ],
                           [ 0.003, 0.0025]])
        inv_sigma1 = np.linalg.inv(sigma1)

        mean2  = np.array( [+5. , 0.   ])
        sigma2 = np.array([[ 0.1, 0.0  ],
                           [ 0.0, 0.5  ]])
        inv_sigma2 = np.linalg.inv(sigma2)

        log_target = lambda x: log( target_abundancies[0] * normalized_pdf_gauss(x, mean1, inv_sigma1) +
                                    target_abundancies[1] * normalized_pdf_gauss(x, mean2, inv_sigma2) )

        # proposal
        prop_abundancies = np.array((.5, .5))

        prop_dof1   = 5.
        prop_mean1  = np.array( [-4.9  , 0.01  ])
        prop_sigma1 = np.array([[ 0.007, 0.0   ],
                                [ 0.0  , 0.0023]])
        prop1       = pmc.proposal.StudentTComponent(prop_mean1, prop_sigma1, prop_dof1)

        prop_dof2   = 5.
        prop_mean2  = np.array( [+5.08, 0.01])
        prop_sigma2 = np.array([[ 0.14, 0.01],
                                [ 0.01, 0.6 ]])
        prop2       = pmc.proposal.StudentTComponent(prop_mean2, prop_sigma2, prop_dof2)

        prop = pmc.proposal.MixtureProposal((prop1, prop2), prop_abundancies)


        sam = pmc.importance_sampling.ImportanceSampler(log_target, prop, rng = np.random.mtrand)
        sam.run(10**4)

        weighted_samples = sam.history[:]
        # -----------------------------------------------------------------------------------------------

        rtol = .05
        atol = .01
        rtol_sigma = .12

        weights = weighted_samples[:,0 ]
        samples = weighted_samples[:,1:]

        clust = GaussianInference(samples, 2, weights=weights, m=np.vstack((prop_mean1,prop_mean2)))
        converged = clust.run(verbose=True)
        self.assertTrue(converged)

        resulting_mixture = clust.get_result()

        sampled_abundancies = resulting_mixture.w
        sampled_mean1       = resulting_mixture.comp[0].mean
        sampled_mean2       = resulting_mixture.comp[1].mean
        sampled_sigma1      = resulting_mixture.comp[0].cov
        sampled_sigma2      = resulting_mixture.comp[1].cov

        np.testing.assert_allclose(sampled_abundancies, target_abundancies, rtol=rtol)
        np.testing.assert_allclose(sampled_mean1[0]   , mean1[0]          , rtol=rtol)
        np.testing.assert_allclose(sampled_mean1[1]   , mean1[1]          , atol=atol) #atol here because target is 0.
        np.testing.assert_allclose(sampled_mean2[0]   , mean2[0]          , rtol=rtol)
        np.testing.assert_allclose(sampled_mean2[1]   , mean2[1]          , atol=atol) #atol here because target is 0.
        np.testing.assert_allclose(sampled_sigma1     , sigma1            , rtol=rtol_sigma)
        np.testing.assert_allclose(sampled_sigma2     , sigma2            , rtol=rtol_sigma, atol=atol) #target is 0. -> atol

    @attr('slow')
    def test_prune(self):
        # generate test data from two independent gaussians
        target_abundances = np.array((.1, .9))
        target_mean1 = np.array((+1. , -4.))
        target_mean2 = np.array((-5. , +2.))
        target_cov1  = covariance1
        target_cov2  = covariance2
        data1 = np.random.mtrand.multivariate_normal(target_mean1, target_cov1, size =   10**2)
        data2 = np.random.mtrand.multivariate_normal(target_mean2, target_cov2, size = 9*10**2)
        test_data = np.vstack((data1,data2))
        np.random.shuffle(test_data)

        # provide hint for means to force convergence to a specific solution
        infer = GaussianInference(test_data, 3, m = np.vstack((target_mean1 +2., target_mean2 + 2., target_mean1+10.)) )
        infer2 = copy.deepcopy(infer)

        nsteps = check_bound(self, infer, 20)
        self.assertTrue(nsteps)
        result = infer.get_result()
        self.assertEqual(len(result.w), 2)
        np.testing.assert_allclose(result[0].mean, target_mean1, rtol=1e-2)
        np.testing.assert_allclose(result[1].mean, target_mean2, rtol=1e-2)

        # run should do the same number of E and M steps
        # to result in same numbers, except it works also when
        # components were reduced => nsteps2 + 1
        eps = 1e-15
        nsteps2 = infer2.run(20, verbose=True)
        self.assertEqual(nsteps2 + 1, nsteps)

        result2 = infer2.get_result()
        np.testing.assert_allclose(result2.w, result.w, rtol=1e-15)
        for i in range(2):
            np.testing.assert_allclose(result2[i].mean, result[i].mean, rtol=1e-15)
            np.testing.assert_allclose(result2[i].cov, result[i].cov, rtol=1e-15)

    # todo test if convergence only approximate

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
        np.random.seed(rng_seed)

    def test_bimodal(self):
        #Compress bimodal distribution with four components to two components.

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

        vb = VBMerge(input_components, N=N, initial_guess=initial_guess, copy_weights=False,
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

        # restart, should converge immediately
        params = vb.prior_posterior()
        vb2 = VBMerge(input_components, **params)
        nsteps = vb2.run(verbose=True)
        self.assertEqual(nsteps, 1)
        self.assertEqual(vb2.likelihood_bound(), vb.likelihood_bound())

        # parameters should be identical at fixed point
        params2 = vb2.prior_posterior()
        for k, v in params2.items():
            np.testing.assert_array_equal(v, params[k])

    @attr('slow')
    def test_large_prune(self):
        #Compress large number of similar components into a single component.

        means = (np.array([5, 0]),)
        cov = np.eye(2)
        N = 500
        N_input = 300
        input_components = create_mixture(means, cov, N_input)

        # first test with only two components to check calculation by hand
        vb_check = VBMerge(input_components, N=N, initial_guess=create_mixture(means, cov, 2), nu=np.zeros(2) + 13.)

        vb_check.likelihood_bound()
        self.assertAlmostEqual(vb_check._expectation_log_p_X, -50014387.38992466, 3)
        self.assertAlmostEqual(vb_check._expectation_log_p_Z, -347.07409, 3)
        self.assertAlmostEqual(vb_check._expectation_log_p_pi, -10.817790168329283)
        self.assertAlmostEqual(vb_check._expectation_log_p_mu_lambda, -54.880566461489643)
        # todo _expectation_log_q_Z unchecked
        self.assertAlmostEqual(vb_check._expectation_log_q_pi, 2.3825149757523718, 6)
        self.assertAlmostEqual(vb_check._expectation_log_q_mu_lambda, -41.029712289)

        # now let lots of components die out
        N_output_initial = 50
        initial_guess = create_mixture(means, cov, N_output_initial)

        nu = np.zeros(N_output_initial) + 3. + 10.

        vb = VBMerge(input_components, N=N, initial_guess=initial_guess, nu=nu)
        vb_prune = copy.deepcopy(vb)
        print('Keep all components...\n')

        self.assertTrue(check_bound(self, vb, prune=False))
        self.assertEqual(vb.K, N_output_initial)

        # now the same thing with pruning, should be faster
        # as only one component remains
        # but bound can decrease if component is removed
        print('Pruning...\n')

        self.assertTrue(check_bound(self, vb_prune, 20))
        self.assertEqual(vb_prune.K, 1)

    def test_bound(self):
        # a Gaussian target, should combine both components
        # taken from MCMC sampling
        target_mean  = np.array([4.3, 1.1])
        target_sigma = np.array([[0.01 , 0.003 ],
                                 [0.003, 0.0025]])
        means  = (np.array([ 4.30733653,  1.10121756]),
                  np.array([ 4.29948   ,  1.09937727]))
        cov   = (np.array([[ 0.01382637,  0.00361037],
                           [ 0.00361037,  0.0043224 ]]),
                 np.array([[ 0.00969403,  0.00292157],
                           [ 0.00292157,  0.00247721]]))
        weights = np.array([ 0.12644431,  0.87355569])
        components = [GaussianMixture.Component(m, c) for m,c in zip(means, cov)]
        input_components = GaussianMixture(components, weights)
        vb = VBMerge(input_components, N=1e4,  components=2)#initial_guess=input_components)
        # compute (43) and (44) manually
        S = np.array([[ 0.01022336,  0.00301026],
                   [ 0.00301026,  0.00271089]])
        np.testing.assert_allclose(vb.S[0], S,  rtol=1e-5)

        # converge exactly in two steps
        # prune out one component after first update
        # then get same bound twice
        self.assertEqual(vb.run(verbose=True), 2)
        self.assertEqual(vb.K, 1)
        res = vb.get_result()
        np.testing.assert_allclose(res[0].mean, target_mean,  rtol=1e-3)
        np.testing.assert_allclose(res[0].cov,  target_sigma, rtol=0.15)

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
