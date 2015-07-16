"""Unit tests for the Gaussian probability densities

"""

from .gauss import *
from .student_t_test import fake_rng
import numpy as np
import unittest

rng_seed = 128501257 % 4294967296
rng_steps = 50000

singular_sigma = np.array([[0.0, 0.0   , 0.0]
                          ,[0.0, 0.0025, 0.0]
                          ,[0.0, 0.0   , 0.6]])

asymmetric_sigma  = np.array([[0.01 , 0.003 ]
                             ,[0.001, 0.0025]])

offdiag_sigma  = np.array([[0.01 , 0.003 ]
                          ,[0.003, 0.0025]])

class TestLocalGauss(unittest.TestCase):
    def test_update(self):
        g = LocalGauss(offdiag_sigma)
        sample = g.propose(np.array([0.,0.]), fake_rng)
        evaluate_at_sample = g.evaluate(np.array([0.,0.]), sample)

        self.assertRaises(np.linalg.LinAlgError, g.update, singular_sigma)
        self.assertRaises(np.linalg.LinAlgError, g.update, asymmetric_sigma)

        # check that the internal variables of ``g`` do not change after LinAlgError
        np.testing.assert_equal(g.sigma, offdiag_sigma)
        self.assertEqual(g.dim, 2)
        np.testing.assert_equal(g.propose (np.array([0.,0.]), fake_rng), sample)
        np.testing.assert_equal(g.evaluate(np.array([0.,0.]), sample), evaluate_at_sample)

    def test_badCovarianceInput(self):
        sigma = singular_sigma
        self.assertRaises(np.linalg.LinAlgError, LocalGauss, (singular_sigma))
        self.assertRaises(np.linalg.LinAlgError, LocalGauss, (asymmetric_sigma))

    def test_evaluate(self):
        sigma = offdiag_sigma
        delta = 1e-8

        t = LocalGauss(sigma=sigma)

        x = np.array([4.3, 1.1])
        y = np.array([4.35, 1.2])

        target = 1.30077135

        self.assertAlmostEqual(t.evaluate(x, y), target, delta=delta)
        self.assertAlmostEqual(t.evaluate(y, x), target, delta=delta)

    def test_propose(self):
        sigma = offdiag_sigma
        delta_chisq = .005
        delta_mean  = .001
        delta_var0  = .0001
        delta_var1  = .00003

        t = LocalGauss(sigma=sigma)

        np.random.seed(rng_seed)
        current = np.array([4.3, 1.1])
        target_mean0 = current[0]
        target_var0  = offdiag_sigma[0,0]
        target_mean1 = current[1]
        target_var1  = offdiag_sigma[1,1]
        target_chisq = 2.0
        log_normalization = -np.log(2. * np.pi) - 0.5 * np.log(1.6e-5)

        values0 = []
        values1 = []
        chisq = 0.

        for i in range(rng_steps-1):
            proposal = t.propose(current, np.random)
            values0 += [proposal[0]]
            values1 += [proposal[1]]
            chisq += -2.0 * (t.evaluate(proposal, current) - log_normalization)

        # test if value for rng can be omitted
        proposal = t.propose(current)
        values0 += [proposal[0]]
        values1 += [proposal[1]]
        chisq += -2.0 * (t.evaluate(proposal, current) - log_normalization)

        chisq /= rng_steps

        values0 = np.array(values0)
        values1 = np.array(values1)

        mean0 = values0.mean()
        mean1 = values1.mean()
        var0  = values0.var()
        var1  = values1.var()


        self.assertAlmostEqual(chisq, target_chisq, delta=delta_chisq)

        self.assertAlmostEqual(mean0, target_mean0, delta=delta_mean)
        self.assertAlmostEqual(mean1, target_mean1, delta=delta_mean)

        self.assertAlmostEqual(var0 , target_var0 , delta=delta_var0)
        self.assertAlmostEqual(var1 , target_var1 , delta=delta_var1)

class TestGauss(unittest.TestCase):
    sigma = np.array([[0.01 , 0.003 ]
                     ,[0.003, 0.0025]])

    mean  = np.array([4.3 , 1.1])
    point = np.array([4.35, 1.2])

    target = 1.30077135

    comp = Gauss(mean, sigma=sigma)

    def setUp(self):
        print('"Gauss" needs .LocalGauss.')
        print('When this test fails, first make sure that .LocalGauss works.')

    def test_update(self):
        g = Gauss(self.mean, offdiag_sigma)
        sample = g.propose(1, fake_rng)[0]
        evaluate_at_sample = g.evaluate(sample)

        self.assertRaises(np.linalg.LinAlgError, g.update, self.point, singular_sigma)
        self.assertRaises(np.linalg.LinAlgError, g.update, self.point, asymmetric_sigma)

        # check that the internal variables of ``g`` do not change after LinAlgError
        np.testing.assert_equal(g.sigma, offdiag_sigma)
        np.testing.assert_equal(g.mu, self.mean)
        self.assertEqual(g.dim, 2)
        np.testing.assert_equal(g.propose(1, fake_rng)[0], sample)
        np.testing.assert_equal(g.evaluate(sample), evaluate_at_sample)

    def test_dim_mismatch(self):
        mu    = np.ones(2)
        sigma = np.eye (3)
        self.assertRaisesRegexp(AssertionError, 'Dimensions of mean \(2\) and covariance matrix \(3\) do not match!', Gauss, mu, sigma)

    def test_evaluate(self):
        self.assertAlmostEqual(self.comp.evaluate(self.point), self.target)

    def test_multi_evaluate(self):
        samples = np.array([self.point] * 2)
        target  = np.array([self.target] * 2)

        out1 = np.empty(2)
        out2 = self.comp.multi_evaluate(samples, out1)
        np.testing.assert_array_almost_equal(out1, target)
        np.testing.assert_array_almost_equal(out2, target)
        assert out1 is out2

        # should also work if out is not provided
        result = self.comp.multi_evaluate(samples)
        np.testing.assert_array_almost_equal(result, target)

    def test_propose(self):
        mean           = np.array([-3.   ,+3.    ])

        offdiag_sigma  = np.array([[0.01 , 0.003 ]
                                  ,[0.003, 0.0025]])

        delta_mean   = .001
        delta_cov_00 = .0001
        delta_cov_01 = .00003
        delta_cov_11 = .00003

        comp = Gauss(mu=mean, sigma=offdiag_sigma)

        np.random.seed(rng_seed)

        # test if value for rng can be omitted
        proposed1 = comp.propose(rng_steps//2)
        # test if value for rng can be set
        proposed2 = comp.propose(rng_steps//2, np.random.mtrand)

        # test standard value for parameter N
        proposed3 = comp.propose()
        self.assertEqual(len(proposed3),1)


        proposed = np.vstack((proposed1, proposed2, proposed3))

        sampled_mean = proposed.mean(axis=0)
        sampled_cov  = np.cov(proposed,rowvar=0)


        self.assertAlmostEqual(sampled_mean[0], mean[0], delta=delta_mean)
        self.assertAlmostEqual(sampled_mean[1], mean[1], delta=delta_mean)

        self.assertAlmostEqual(sampled_cov[0,0] , offdiag_sigma[0,0] , delta=delta_cov_00)
        self.assertAlmostEqual(sampled_cov[0,1] , offdiag_sigma[0,1] , delta=delta_cov_01)
        self.assertAlmostEqual(sampled_cov[1,1] , offdiag_sigma[1,1] , delta=delta_cov_11)
