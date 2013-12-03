"""Unit tests for the MCMC sampler functions.

"""

from markov_chain import *
import proposal
import numpy as np
import unittest

offdiagSigma  = np.array([[0.01 , 0.003 ]
                         ,[0.003, 0.0025]])

rng_seed = 215135153

NumberOfRandomSteps = 50000

def raise_not_implemented(x):
    raise NotImplementedError()

class MultivariateNonEvaluable(proposal.MultivariateGaussian):
    def evaluate(self, x, y):
        raise NotImplementedError()

def unnormalized_log_pdf_gauss(x, mu, inv_sigma):
    return - .5 * np.dot(np.dot(x-mu, inv_sigma), x-mu)

class TestMarkovChain(unittest.TestCase):
    def test_indicator(self):
        prop = proposal.MultivariateGaussian(offdiagSigma)
        indicator = lambda x: False
        start = np.array((0.,1.))

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

        mc = MarkovChain(log_target, prop, start)

        #prerun for burn-in
        mc.run(NumberOfRandomSteps/10)
        self.assertEqual(len(mc.points), NumberOfRandomSteps/10+1)
        mc.clear()
        self.assertEqual(len(mc.points), 1)

        mc.run(NumberOfRandomSteps)

        values = np.array(mc.points)

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
