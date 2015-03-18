"""Unit tests for the MCMC sampler functions.

"""

from .markov_chain import *
from .. import density
from ..tools._probability_densities import unnormalized_log_pdf_gauss
from nose.plugins.attrib import attr
import numpy as np
import unittest

zero_mean      = np.zeros(2)

offdiag_sigma  = np.array([[0.01 , 0.003 ]
                          ,[0.003, 0.0025]])

rng_seed = 215135153

NumberOfRandomSteps = 50000

class RejectAllRNG(object):
    def normal(self, a, b, N):
        return np.array(N*[1.])
    def chisquare(self, degree_of_freedom):
        # print 'in FakeRNG: degree_of_freedom', degree_of_freedom
        assert type(degree_of_freedom) == float
        return degree_of_freedom
    def rand(self):
        # this makes the Markov chain reject every sample
        return 1.
reject_rng = RejectAllRNG()

def raise_not_implemented(x):
    if (x == np.array((0.,1.))).all():
        return 1.
    raise NotImplementedError()

def infinite(x):
    return np.inf

def nan(x):
    return np.nan

class MultivariateNonEvaluable(density.gauss.LocalGauss):
    def evaluate(self, x, y):
        raise NotImplementedError()

class TestMarkovChain(unittest.TestCase):
    def setUp(self):
        np.random.mtrand.seed(rng_seed)

    def test_copy(self):
        prop = density.gauss.LocalGauss(offdiag_sigma)

        mc = MarkovChain(lambda x: 1., prop, zero_mean)

        self.assertNotEqual(id(mc.current_point), id(zero_mean), msg='starting point has not been copied')
        self.assertNotEqual(id(mc.proposal)     , id(prop     ), msg='Proposal has not been copied')

    def test_invalid_start(self):
        prop = density.gauss.LocalGauss(offdiag_sigma)
        start = np.array((0.,1.))

        with self.assertRaises(ValueError):
            mc = MarkovChain(nan, prop, start)

        with self.assertRaises(ValueError):
            mc = MarkovChain(infinite, prop, start)

    def test_indicator(self):
        prop = density.gauss.LocalGauss(offdiag_sigma)
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
        prop = MultivariateNonEvaluable(offdiag_sigma)
        start = np.array((0.,1.))

        mc = MarkovChain(lambda x: 1., prop, start)

        mc.run()

        self.assertRaises(NotImplementedError, lambda: prop.evaluate(1.,2.))

    @attr('slow')
    def test_sampling(self):
        delta_mean   = .002
        delta_var0   = .0003
        delta_var1   = .00003

        prop_dof   = 5.
        prop_sigma = np.array([[0.1 , 0.  ]
                               ,[0.  , 0.02]])

        prop = density.student_t.LocalStudentT(prop_sigma, prop_dof)

        target_sigma = offdiag_sigma
        target_mean  = np.array([4.3, 1.1])
        log_target = lambda x: unnormalized_log_pdf_gauss(x, target_mean, np.linalg.inv(offdiag_sigma))

        #extremely bad starting values
        start = np.array([-3.7, 10.6])

        mc = MarkovChain(log_target, prop, start, prealloc = NumberOfRandomSteps)

        #prerun for burn-in
        mc.run(int(NumberOfRandomSteps/10))
        self.assertEqual(len(mc.samples[-1]), NumberOfRandomSteps//10)
        mc.clear()
        self.assertEqual(len(mc.samples), 0)

        mc.run(NumberOfRandomSteps)

        values = mc.samples[:]

        mean0 = values[:,0].mean()
        mean1 = values[:,1].mean()
        var0  = values[:,0].var()
        var1  = values[:,1].var()


        self.assertAlmostEqual(mean0, target_mean[0]   , delta=delta_mean)
        self.assertAlmostEqual(mean1, target_mean[1]   , delta=delta_mean)

        self.assertAlmostEqual(var0 , target_sigma[0,0], delta=delta_var0)
        self.assertAlmostEqual(var1 , target_sigma[1,1], delta=delta_var1)

    def test_run_notices_NaN(self):
        bad_target = lambda x: 0. if (x==start).all() else np.nan
        prop       = density.gauss.LocalGauss(offdiag_sigma)
        start      = np.array([4.3, 1.1])

        mc = MarkovChain(bad_target, prop, start)

        self.assertRaisesRegexp(ValueError, 'encountered NaN', mc.run)

    def test_history(self):
        # dummy; not a real proposal
        class ProposalPlusOne(density.base.ProbabilityDensity):
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
        mc = MarkovChain(target, ProposalPlusOne(), start, prealloc=50)

        # the above configuration creates an ever accepting markov chain
        # the visited points will be 0., 1., 2., 3., 4., ...
        # this is convenient when testing if the history is stored correctly

        for i in range(10):
            mc.run(10)

        float_acc = 1e-15

        # check runs
        for run in range(10):
            this_run        = mc.samples[run]
            self.assertEqual(len(this_run), 10)
            this_run_target = np.arange(1.,11.)+(run)*10
            for i in range(10):
                self.assertAlmostEqual(this_run[i][0], this_run_target[i], float_acc)

        # check slicing
        for i in range(50):
            self.assertAlmostEqual(mc.samples[:6][i], np.arange(1,51)[i], float_acc)

    def test_save_target_values(self):
        N = 10

        prop_sigma = np.array([[0.1 , 0.  ]
                               ,[0.  , 0.02]])

        prop = density.student_t.LocalGauss(prop_sigma)

        target_sigma = offdiag_sigma
        target_mean  = np.array([4.3, 1.1])
        log_target = lambda x: unnormalized_log_pdf_gauss(x, target_mean, np.linalg.inv(offdiag_sigma))

        start = np.array([-3.7, 10.6])

        mc = MarkovChain(log_target, prop, start, prealloc=N)
        self.assertEqual(mc.target_values, None)
        mc = MarkovChain(log_target, prop, start, prealloc=N, save_target_values=True)

        mc.run(N)
        self.assertEqual(len(mc.samples[-1]), N)
        samples       = mc.samples[:]
        target_values = mc.target_values[:]

        self.assertEqual(len(mc.target_values), 1)
        self.assertEqual(len(mc.samples), 1)

        self.assertEqual(len(mc.target_values[:]), N)
        self.assertEqual(len(mc.samples[:]), N)

        for i in range(10):
            # check if target values are correctly saved
            self.assertEqual(log_target(mc.samples[:][i]), mc.target_values[:][i])

        mc.clear()

        self.assertEqual(len(mc.target_values), 0)
        self.assertEqual(len(mc.samples), 0)

class TestAdaptiveMarkovChain(unittest.TestCase):
    def setUp(self):
        np.random.mtrand.seed(rng_seed)

    @attr('slow')
    def test_adapt(self):
        delta_mean   = .005

        relative_error_unscaled_sigma = .05

        prop_dof   = 50.
        prop_sigma = np.array([[0.1 , 0.  ]
                               ,[0.  , 0.02]])

        prop = density.student_t.LocalStudentT(prop_sigma, prop_dof)

        target_sigma = offdiag_sigma
        target_mean  = np.array([4.3, 1.1])
        log_target = lambda x: unnormalized_log_pdf_gauss(x, target_mean, np.linalg.inv(offdiag_sigma))

        #good starting values; prerun is already tested in TestMarkovChain
        start = np.array([4.2, 1.])

        mc = AdaptiveMarkovChain(log_target, prop, start, prealloc = NumberOfRandomSteps)

        scale_up_visited   = False
        scale_down_visited = False
        covar_scale_factor = 1.
        mc.set_adapt_params(covar_scale_factor = covar_scale_factor)

        for i in range(10):
            mc.run(NumberOfRandomSteps//10)
            mc.adapt()

            if   mc.covar_scale_factor > covar_scale_factor:
                scale_up_visited   = True
            elif mc.covar_scale_factor < covar_scale_factor:
                scale_down_visited = True

            covar_scale_factor = mc.covar_scale_factor

        values = mc.samples[:]

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

    def test_enhanced_adapt(self):
        # test the behavior of ``.adapt`` if the standard approach fails
        dim = 3
        initial_prop_sigma = np.array([[0.1 , 0.03, 0.01]
                                      ,[0.03, 0.02, 0.0 ]
                                      ,[0.01, 0.0 , 1.0 ]])
        prop = density.gauss.LocalGauss(initial_prop_sigma)
        start1 = (1,1,1)
        start2 = (0,0,0)
        log_target = lambda x: 1. if (x == (0,0,0)).all() else 0.

        # check that the diagonalized matrix is used if the standard approach fails
        # -------------------------------------------------------------------------

        mc = AdaptiveMarkovChain(log_target, prop, start1)

        # in three dimensions, the covariance matrix estimated from two samples
        # is neccessarily singular
        accept_count = mc.run(2)

        # make sure that both samples are accepted to yield valid variance etimates
        self.assertEqual(accept_count, 2)

        mc.adapt()

        target_unscaled_sigma = np.cov(mc.samples[-1], rowvar=0)
        target_covar_scale_factor = 2.38**2 / dim * 1.5
        target_proposal_sigma = np.diag(np.diag(mc.covar_scale_factor * mc.unscaled_sigma))

        np.testing.assert_equal(mc.unscaled_sigma, target_unscaled_sigma)
        self.assertAlmostEqual(mc.covar_scale_factor, target_covar_scale_factor)
        np.testing.assert_almost_equal(mc.proposal.sigma, target_proposal_sigma, decimal=13)

        # check that the covariance is scaled if the diagonalization approach fails
        # -------------------------------------------------------------------------

        # reject_rng and start at (0,0,0) --> reject every sample to make sure that
        # even the diagonalized covariance matrix is singular
        mc = AdaptiveMarkovChain(log_target, prop, start2, rng=reject_rng)

        accept_count = mc.run(10)
        self.assertEqual(accept_count, 0)

        mc.adapt()

        target_unscaled_sigma = np.zeros((3,3))
        target_covar_scale_factor = 2.38**2 / dim / 1.5
        target_proposal_sigma = initial_prop_sigma / 1.5

        np.testing.assert_equal(mc.unscaled_sigma, target_unscaled_sigma)
        self.assertAlmostEqual(mc.covar_scale_factor, target_covar_scale_factor)
        np.testing.assert_almost_equal(mc.proposal.sigma, target_proposal_sigma, decimal=13)

    def test_set_adapt_parameters(self):
        log_target = lambda x: unnormalized_log_pdf_gauss(x, zero_mean, np.linalg.inv(offdiag_sigma))
        prop_sigma = np.array([[1. , 0.  ]
                              ,[0. , 1.  ]])
        prop = density.gauss.LocalGauss(prop_sigma)
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
