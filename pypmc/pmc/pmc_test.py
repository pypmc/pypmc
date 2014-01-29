'''Unit tests for the Population Monte Carlo.

'''

from .pmc import *
from . import proposal
from .._tools._probability_densities import unnormalized_log_pdf_gauss
from math import exp
import numpy as np
import unittest

rng_seed  = 295627184
rng_steps = 50000

dim = 2

mu  = np.array( [1.     , -1.   ] )

cov = np.array([[11.5   ,  2.875],
                [2.875  ,  0.75 ]])

inv_cov = np.linalg.inv(cov)

class TestPMC(unittest.TestCase):
    def setUp(self):
        np.random.mtrand.seed(rng_seed)

        self.log_target = lambda x: unnormalized_log_pdf_gauss(x, mu, inv_cov)

    def test_run(self):
        perfect_prop = proposal.MixtureProposal((proposal.GaussianComponent(mu, cov),))

        # perturbed proposal
        initial_prop = proposal.MixtureProposal((proposal.GaussianComponent(mu+.1, cov+.1),))

        pmc = GaussianPMC(self.log_target, initial_prop, rng=np.random.mtrand)

        pmc.run(rng_steps-1)
        samples_weights_first_step = pmc.hist[:][1].copy() # need a copy because weights will be overwritten
        pmc.proposal.components[0].update(mu, cov) # set proposal = normalized target (i.e. perfect_prop)
        pmc.run(rng_steps)
        samples_weights_second_step = pmc.hist[:2][1]

        # first column is weight -> cut it to get samples only
        samples_first  = samples_weights_first_step [:,1:]
        samples_second = samples_weights_second_step[:,1:]

        weights_first  = samples_weights_first_step [:,0]
        weights_second = samples_weights_second_step[:,0]

        # samples should be unchanged (bitwise equal), only the weights should differ
        for i in range(rng_steps):
            for j in range(dim):
                self.assertEqual(samples_first[i,j], samples_second[i,j])

        # check weights before adaptation
        for i, sample in enumerate(samples_first):
            target = exp(self.log_target(sample)) / exp(initial_prop.evaluate(sample))
            self.assertAlmostEqual(weights_first[i], target)

        # check weights after adaptation
        for i, sample in enumerate(samples_second):
            # simplify (3) in [Cor+12]
            target = 2.*exp(self.log_target(sample)) / ( exp(initial_prop.evaluate(sample)) + exp(perfect_prop.evaluate(sample)) )
            self.assertAlmostEqual(weights_second[i], target)

    def test_adapt(self):
        delta_weights = .001
        delta_mean    = .05
        delta_cov     = .05

        # perturbed
        gauss0 = proposal.GaussianComponent(mu+.1  , cov+1. )
        # far away
        gauss1 = proposal.GaussianComponent(mu+100., cov+100.)

        initial_prop = proposal.MixtureProposal((gauss0,gauss1))

        pmc = GaussianPMC(self.log_target, initial_prop, rng=np.random.mtrand)

        for i in range(5):
            pmc.run(rng_steps//5)
            pmc.adapt()
            self.assertTrue(pmc.proposal.normalized())

        target_weights  = (1., 0.)
        target_mean0    = mu
        target_cov0     = cov

        adapted_weights = pmc.proposal.weights
        adapted_mean0   = pmc.proposal.components[0].mu
        adapted_cov0    = pmc.proposal.components[0].sigma

        self.assertAlmostEqual    (adapted_weights[0], target_weights[0], delta=delta_weights)
        self.assertAlmostEqual    (adapted_weights[1], target_weights[1], delta=delta_weights)
        np.testing.assert_allclose(adapted_mean0     , target_mean0     , rtol=delta_mean    )
        np.testing.assert_allclose(adapted_cov0      , target_cov0      , rtol=delta_cov     )

    def test_clear(self):
        prop = proposal.MixtureProposal((proposal.GaussianComponent(mu, cov),))
        pmc = GaussianPMC(self.log_target, prop, rng=np.random.mtrand)
        pmc.run(10)
        pmc.hist.clear()
        self.assertRaisesRegexp(AssertionError, r'^Inconsistent state(.*)try ["\'`]*self.clear', pmc.run)
        pmc.clear()
        pmc.run(19)
        N, weighted_samples = pmc.hist[:]
        self.assertEqual(N, 20)
        self.assertEqual(len(weighted_samples), 20)
