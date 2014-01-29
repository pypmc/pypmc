"""Unit tests for the MCMC proposal functions.

"""

from .proposal import *
import numpy as np
import unittest

rng_seed = 128501257
NumberOfRandomSteps = 50000

singularSigma = np.array([[0.0, 0.0   , 0.0]
                         ,[0.0, 0.0025, 0.0]
                         ,[0.0, 0.0   , 0.6]])

offdiagSigma  = np.array([[0.01 , 0.003 ]
                         ,[0.003, 0.0025]])

class TestGaussian(unittest.TestCase):
    def test_badCovarianceInput(self):
        sigma = singularSigma
        self.assertRaises(np.linalg.LinAlgError, MultivariateGaussian, (sigma))

    def test_evaluate(self):
        sigma = offdiagSigma
        delta = 1e-8

        t = MultivariateGaussian(sigma=sigma)

        x = np.array([4.3, 1.1])
        y = np.array([4.35, 1.2])

        target = 1.30077135

        self.assertAlmostEqual(t.evaluate(x, y), target, delta=delta)
        self.assertAlmostEqual(t.evaluate(y, x), target, delta=delta)

    def test_propose(self):
        sigma = offdiagSigma
        delta_chisq = .005
        delta_mean  = .001
        delta_var0  = .0001
        delta_var1  = .00003

        t = MultivariateGaussian(sigma=sigma)

        np.random.seed(rng_seed)
        current = np.array([4.3, 1.1])
        target_mean0 = current[0]
        target_var0  = offdiagSigma[0,0]
        target_mean1 = current[1]
        target_var1  = offdiagSigma[1,1]
        target_chisq = 2.0
        log_normalization = -np.log(2. * np.pi) - 0.5 * np.log(1.6e-5)

        values0 = []
        values1 = []
        chisq = 0.

        for i in range(NumberOfRandomSteps-1):
            proposal = t.propose(current, np.random)
            values0 += [proposal[0]]
            values1 += [proposal[1]]
            chisq += -2.0 * (t.evaluate(proposal, current) - log_normalization)

        # test if value for rng can be omitted
        proposal = t.propose(current)
        values0 += [proposal[0]]
        values1 += [proposal[1]]
        chisq += -2.0 * (t.evaluate(proposal, current) - log_normalization)

        chisq /= NumberOfRandomSteps

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

class TestStudentT(unittest.TestCase):
    def test_badCovarianceInput(self):
        sigma = singularSigma
        self.assertRaises(np.linalg.LinAlgError, MultivariateGaussian, (sigma))

    def test_evaluate(self):
        sigma = np.array([[0.0049, 0.  ]
                         ,[0.    ,  .01]])
        delta = 1e-9
        dof   = 5.

        t = MultivariateStudentT(sigma=sigma, dof=dof)

        x0 = np.array([1.25, 4.3  ])
        x1 = np.array([1.3 , 4.4  ])
        x2 = np.array([1.26, 4.424])

        target1 = 2.200202941
        target2 = 2.174596526

        self.assertAlmostEqual(t.evaluate(x1, x0), target1, delta=delta)
        self.assertAlmostEqual(t.evaluate(x0, x1), target1, delta=delta)

        self.assertAlmostEqual(t.evaluate(x2, x0), target2, delta=delta)
        self.assertAlmostEqual(t.evaluate(x0, x2), target2, delta=delta)

    def test_propose_covariance(self):
        sigma = np.array([[.2]])
        dof   = 5.
        delta = 0.005

        t = MultivariateStudentT(sigma = sigma, dof = dof)

        np.random.seed(rng_seed)
        current = np.array([0.])
        target_mean = current[0]
        target_var  = dof / (dof - 2.) * sigma[0,0]

        mean = 0.
        var  = 0.

        for i in range(NumberOfRandomSteps-1):
            proposal = t.propose(current, np.random)
            mean += proposal[0]
            var  += proposal[0]**2

        # test if value for rng can be omitted
        proposal = t.propose(current)
        mean += proposal[0]
        var  += proposal[0]**2

        mean /= NumberOfRandomSteps
        var  /= NumberOfRandomSteps
        var  -= mean**2

        self.assertAlmostEqual(mean, target_mean, delta=delta)
        self.assertAlmostEqual(var , target_var , delta=delta)

    def test_propose_2D(self):
        sigma      = offdiagSigma
        dof        = 5.
        delta_mean = .001
        delta_var0 = .0001
        delta_var1 = .00003

        t = MultivariateStudentT(sigma=sigma, dof=dof)

        np.random.seed(rng_seed)
        current = np.array([4.3, 1.1])
        target_mean0 = current[0]
        target_var0  = offdiagSigma[0,0] * dof/(dof-2)
        target_mean1 = current[1]
        target_var1  = offdiagSigma[1,1] * dof/(dof-2)

        values0 = []
        values1 = []

        for i in range(NumberOfRandomSteps-1):
            proposal = t.propose(current, np.random)
            values0 += [proposal[0]]
            values1 += [proposal[1]]

        # test if value for rng can be omitted
        proposal = t.propose(current)
        values0 += [proposal[0]]
        values1 += [proposal[1]]

        values0 = np.array(values0)
        values1 = np.array(values1)

        mean0 = values0.mean()
        var0  = values0.var()
        mean1 = values1.mean()
        var1  = values1.var()

        self.assertAlmostEqual(mean0, target_mean0, delta=delta_mean)
        self.assertAlmostEqual(var0 , target_var0 , delta=delta_var0)
        self.assertAlmostEqual(mean1, target_mean1, delta=delta_mean)
        self.assertAlmostEqual(var1 , target_var1 , delta=delta_var1)
