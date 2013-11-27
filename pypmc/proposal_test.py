"""Unit tests for the MCMC proposal functions.

"""

from proposal import *
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
        delta = .001

        t = MultivariateGaussian(sigma=sigma)

        np.random.seed(rng_seed)
        current = np.array([4.3, 1.1])
        target_mean1 = 4.3
        target_mean2 = 1.1
        target_chiSq = 2.0
        normalization = -np.log(2. * np.pi) - 0.5 * np.log(1.6e-5)

        mean1 = 0.
        mean2 = 0.
        chiSq = 0.

        for i in xrange(NumberOfRandomSteps-1):
            proposal = t.propose(current, np.random)
            mean1 += proposal[0]
            mean2 += proposal[1]
            chiSq += -2.0 * (t.evaluate(proposal, current) - normalization)
        proposal = t.propose(current)
        mean1 += proposal[0]
        mean2 += proposal[1]
        chiSq += -2.0 * (t.evaluate(proposal, current) - normalization)

        mean1 /= NumberOfRandomSteps
        mean2 /= NumberOfRandomSteps
        chiSq /= NumberOfRandomSteps

        self.assertAlmostEqual(mean1, target_mean1, delta=.001)
        self.assertAlmostEqual(mean2, target_mean2, delta=.001)
        self.assertAlmostEqual(chiSq, target_chiSq, delta=.005)

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

        for i in xrange(NumberOfRandomSteps-1):
            proposal = t.propose(current, np.random)
            mean += proposal[0]
            var  += proposal[0]**2
        proposal = t.propose(current)
        mean += proposal[0]
        var  += proposal[0]**2

        mean /= NumberOfRandomSteps
        var  /= NumberOfRandomSteps
        var  -= mean**2

        self.assertAlmostEqual(mean, target_mean, delta=delta)
        self.assertAlmostEqual(var , target_var , delta=delta)

    def test_propose_2D(self):
        sigma = offdiagSigma
        dof   = 5.
        delta = .001

        t = MultivariateStudentT(sigma=sigma, dof=dof)

        np.random.seed(rng_seed)
        current = np.array([4.3, 1.1])
        target_mean1 = 4.3
        target_mean2 = 1.1

        mean1 = 0.
        mean2 = 0.

        for i in xrange(NumberOfRandomSteps-1):
            proposal = t.propose(current, np.random)
            mean1 += proposal[0]
            mean2 += proposal[1]
        proposal = t.propose(current)
        mean1 += proposal[0]
        mean2 += proposal[1]

        mean1 /= NumberOfRandomSteps
        mean2 /= NumberOfRandomSteps

        self.assertAlmostEqual(mean1, target_mean1, delta=delta)
        self.assertAlmostEqual(mean2, target_mean2, delta=delta)
