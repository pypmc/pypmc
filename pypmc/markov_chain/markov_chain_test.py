"""Unit tests for the MCMC sampler functions.

"""

from .markov_chain import *
from . import proposal
import numpy as np
import unittest

offdiagSigma  = np.array([[0.01 , 0.003 ]
                         ,[0.003, 0.0025]])

rng_seed = 215135153

NumberOfRandomSteps = 50000

def raise_not_implemented(x):
    if (x == np.array((0.,1.))).all():
        return 1.
    raise NotImplementedError()

class MultivariateNonEvaluable(proposal.MultivariateGaussian):
    def evaluate(self, x, y):
        raise NotImplementedError()

def unnormalized_log_pdf_gauss(x, mu, inv_sigma):
    return - .5 * np.dot(np.dot(x-mu, inv_sigma), x-mu)

class TestMarkovChain(unittest.TestCase):
    def test_indicator(self):
        prop = proposal.MultivariateGaussian(offdiagSigma)
        start = np.array((0.,1.))
        indicator = lambda x: (x == start).all()

        mc_with_ind = MarkovChain(raise_not_implemented, prop, start, indicator)

        #explicitly missinig the indicator argument to check if standard value works
        mc_no_ind   = MarkovChain(raise_not_implemented, prop, start)

        self.assertRaises(NotImplementedError, mc_no_ind.run)
        #explicitly missing arguments to check standard values
        mc_with_ind.run()

    def test_symmetric(self):
        # TODO: extend this test to sample from non-symmetric proposal

        # proposal.evaluate should never be called if proposal.symmetric == True
        prop = MultivariateNonEvaluable(offdiagSigma)
        start = np.array((0.,1.))

        np.random.mtrand.seed(rng_seed)

        mc = MarkovChain(lambda x: 1., prop, start)

        mc.run()

        self.assertRaises(NotImplementedError, lambda: prop.evaluate(1.,2.))

    def test_sampling(self):
        delta_mean   = .002
        delta_var0   = .0003
        delta_var1   = .00003

        prop_dof   = 5.
        prop_sigma = np.array([[0.1 , 0.  ]
                               ,[0.  , 0.02]])

        prop = proposal.MultivariateStudentT(prop_sigma, prop_dof)

        target_sigma = offdiagSigma
        target_mean  = np.array([4.3, 1.1])
        log_target = lambda x: unnormalized_log_pdf_gauss(x, target_mean, np.linalg.inv(offdiagSigma))

        #extremely bad starting values
        start = np.array([-3.7, 10.6])

        np.random.mtrand.seed(rng_seed)

        mc = MarkovChain(log_target, prop, start, prealloc = NumberOfRandomSteps)

        #prerun for burn-in
        mc.run(int(NumberOfRandomSteps/10))
        self.assertEqual(len(mc.hist[-1][1]), int(NumberOfRandomSteps/10))
        mc.hist.clear()
        self.assertEqual(len(mc.hist[-1][1]), 1)

        mc.run(NumberOfRandomSteps)

        values = mc.hist[:][1]

        mean0 = values[:,0].mean()
        mean1 = values[:,1].mean()
        var0  = values[:,0].var()
        var1  = values[:,1].var()


        self.assertAlmostEqual(mean0, target_mean[0]   , delta=delta_mean)
        self.assertAlmostEqual(mean1, target_mean[1]   , delta=delta_mean)

        self.assertAlmostEqual(var0 , target_sigma[0,0], delta=delta_var0)
        self.assertAlmostEqual(var1 , target_sigma[1,1], delta=delta_var1)

    def test_run_notices_NaN(self):
        bad_target = lambda x: np.nan
        prop       = proposal.MultivariateGaussian(offdiagSigma)
        start      = np.array([4.3, 1.1])

        mc = MarkovChain(bad_target, prop, start)

        self.assertRaisesRegexp(ValueError, 'encountered NaN', mc.run)

    def test_hist(self):
        # dummy; not a real proposal
        class ProposalPlusOne(proposal.ProposalDensity):
            symmetric = True
            def __init__(self):
                pass
            def propose(self, y, rng):
                return y + 1.

        # dummy target
        def target(x):
            return x + 1

        start = np.array((0.,))

        # preallocate for half of the planned sampling length to check both,
        # use of preallocated memory and dynamically reallocated memory
        mc = MarkovChain(target, ProposalPlusOne(), start, prealloc = 50)

        # the above configuration creates an ever accepting markov chain
        # the visited points will be 0., 1., 2., 3., 4., ...
        # this is convenient when testing if the history is stored correctly

        for i in range(10):
            mc.run(10)

        float_acc = 1e-15

        # check initial point
        self.assertAlmostEqual(0., mc.hist[0][1])

        # check all runs
        for run in range(1,11):
            this_run        = mc.hist[run][1]
            this_run_target = np.arange(1.,11.)+(run-1)*10
            for i in range(10):
                self.assertAlmostEqual(this_run[i][0], this_run_target[i], float_acc)

        # check slicing
        for i in range(50):
            self.assertAlmostEqual(mc.hist[:6][1][i], np.arange(50)[i], float_acc)

class TestAdaptiveMarkovChain(unittest.TestCase):
    def test_adapt(self):
        delta_mean   = .005

        relative_error_unscaled_sigma = .05

        prop_dof   = 50.
        prop_sigma = np.array([[0.1 , 0.  ]
                               ,[0.  , 0.02]])

        prop = proposal.MultivariateStudentT(prop_sigma, prop_dof)

        target_sigma = offdiagSigma
        target_mean  = np.array([4.3, 1.1])
        log_target = lambda x: unnormalized_log_pdf_gauss(x, target_mean, np.linalg.inv(offdiagSigma))

        #good starting values; prerun is already tested in TestMarkovChain
        start = np.array([4.2, 1.])

        np.random.mtrand.seed(rng_seed)

        mc = AdaptiveMarkovChain(log_target, prop, start, prealloc = NumberOfRandomSteps)

        scale_up_visited   = False
        scale_down_visited = False
        covar_scale_factor = 1.
        mc.set_adapt_params(covar_scale_factor = covar_scale_factor)

        for i in range(10):
            mc.run(int(NumberOfRandomSteps/10))
            mc.adapt()

            if   mc.covar_scale_factor > covar_scale_factor:
                scale_up_visited   = True
            elif mc.covar_scale_factor < covar_scale_factor:
                scale_down_visited = True

            covar_scale_factor = mc.covar_scale_factor

        values = mc.hist[:][1]

        mean0 = values[:,0].mean()
        mean1 = values[:,1].mean()

        self.assertAlmostEqual(mean0, target_mean[0]   , delta=delta_mean)
        self.assertAlmostEqual(mean1, target_mean[1]   , delta=delta_mean)

        self.assertAlmostEqual(mc.unscaled_sigma[0,0] , target_sigma[0,0], delta=relative_error_unscaled_sigma * target_sigma[0,0])
        self.assertAlmostEqual(mc.unscaled_sigma[0,1] , target_sigma[0,1], delta=relative_error_unscaled_sigma * target_sigma[0,1])
        self.assertAlmostEqual(mc.unscaled_sigma[1,0] , target_sigma[1,0], delta=relative_error_unscaled_sigma * target_sigma[1,0])
        self.assertAlmostEqual(mc.unscaled_sigma[1,1] , target_sigma[1,1], delta=relative_error_unscaled_sigma * target_sigma[1,1])

        self.assertTrue(scale_up_visited)
        self.assertTrue(scale_down_visited)

    def test_set_adapt_parameters(self):
        log_target = lambda x: unnormalized_log_pdf_gauss(x, target_mean, np.linalg.inv(offdiagSigma))
        prop_sigma = np.array([[1. , 0.  ]
                              ,[0. , 1.  ]])
        prop = proposal.MultivariateGaussian(prop_sigma)
        start = np.array([4.2, 1.])

        test_value = 4.

        mc = AdaptiveMarkovChain(log_target, prop, start, covar_scale_multiplier = test_value)

        mc.set_adapt_params(                                     covar_scale_factor     = test_value,
                            covar_scale_factor_max = test_value, covar_scale_factor_min = test_value)

        mc.set_adapt_params(force_acceptance_max   = test_value, force_acceptance_min   = test_value)

        mc.set_adapt_params(damping                = test_value)

        self.assertEqual(mc.covar_scale_multiplier , test_value)
        self.assertEqual(mc.covar_scale_factor     , test_value)
        self.assertEqual(mc.covar_scale_factor_max , test_value)
        self.assertEqual(mc.covar_scale_factor_min , test_value)
        self.assertEqual(mc.force_acceptance_max   , test_value)
        self.assertEqual(mc.force_acceptance_min   , test_value)
        self.assertEqual(mc.damping                , test_value)

        self.assertRaisesRegexp(TypeError, r'keyword args only; try set_adapt_parameters\(keyword = value\)',
                                mc.set_adapt_params, test_value)
        self.assertRaisesRegexp(TypeError, r"unexpected keyword\(s\)\: ",
                                mc.set_adapt_params, unknown_kw = test_value)
