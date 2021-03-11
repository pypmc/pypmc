"""Unit tests for the hierarchical clustering.

"""
from .hierarchical import *
from ..density.gauss import Gauss
from ..density.mixture import MixtureDensity
import numpy as np
import unittest

hierarchical_cov = np.eye(2) #must be defined outside for python3 --> see line 'initial_components = ...'

class TestHierarchical(unittest.TestCase):
    # bimodal Gaussian distribution with a few components
    # randomly scattered around each mode
    means = [np.array([+5, 0]), np.array([-5, 0])]
    cov = hierarchical_cov

    # generate 50% of components from each mode
    ncomp = 500
    np.random.seed(ncomp)
    random_centers1 = np.random.multivariate_normal(means[0], cov, size=(ncomp // 2))
    random_centers2 = np.vstack((random_centers1, np.random.multivariate_normal(means[1], cov, size=ncomp // 2)))
    #input_components = GaussianMixture([GaussianMixture.Component(mu, cov) for mu in random_centers]) --> does not work in python3
    input_components1 = MixtureDensity([Gauss(mu, hierarchical_cov) for mu in random_centers1])
    input_components2 = MixtureDensity([Gauss(mu, hierarchical_cov) for mu in random_centers2])
    initial_guess1 = MixtureDensity([Gauss(np.zeros(2) + 1e10, cov*3),
                      Gauss(np.zeros(2) - 0.1, cov*3),])
    initial_guess2 = MixtureDensity([Gauss(np.zeros(2) + 0.1, cov*3), Gauss(np.zeros(2) - 0.1, cov*3),])

    def test_prune(self):
        h = Hierarchical(self.input_components1, self.initial_guess1)
        h.run(verbose=True)
        sol = h.g

        # only one component should survive and have weight 1.0
        # expect precision loss in the weight summation for many input components
        self.assertEqual(len(sol.components), 1)
        self.assertAlmostEqual(sol.weights[0], 1., 14)

        # means should be reproduced
        eps = 12e-2
        self.assertAlmostEqual(sol.components[0].mu[0], self.means[0][0], delta=eps)

        # variance much larger now, but still should have little correlation
        self.assertAlmostEqual(sol.components[0].sigma[0,1], 0, delta=0.1)

    def test_cluster(self):
        h = Hierarchical(self.input_components2, self.initial_guess2)
        h.run(verbose=True)
        sol = h.g

        # both components should survive and have equal weight
        # expect precision loss in the weight summation for many input components
        self.assertEqual(len(sol.components), 2)
        self.assertAlmostEqual(sol.weights[0], 0.5, 13)

        # means should be reproduced
        eps = 12e-2
        self.assertAlmostEqual(sol.components[0].mu[0], self.means[0][0], delta=eps)
        self.assertAlmostEqual(sol.components[1].mu[0], self.means[1][0], delta=eps)

        # variance much larger now, but still should have little correlation
        for i in range(2):
            self.assertAlmostEqual(sol.components[i].sigma[0,1], 0, delta=0.1)

class TestKL(unittest.TestCase):
    def test_2D(self):
        d = 2
        mu1 = np.array([1, 3])
        mu2 = np.zeros(d)
        cov = np.eye(d)

        c1 = Gauss(mu1, cov)
        c2 = Gauss(mu2, cov)

        self.assertAlmostEqual(kullback_leibler(c1, c2), 0.5 * mu1.dot(mu1), 15)

        # now add correlation, matrix inverse not trivial anymore
        cov2 = np.eye(d)
        cov2[0,1] = cov2[1,0] = 0.2
        c2 = Gauss(mu2, cov2)
        self.assertAlmostEqual(kullback_leibler(c1, c2), 4.6045890027398722, 15)
