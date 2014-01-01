"""Unit tests for the hierarchical clustering.

"""
from __future__ import division
from .hierarchical import *
from .gaussian_mixture import GaussianMixture
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
    random_centers = np.random.multivariate_normal(means[0], cov, size=(ncomp // 2))
    random_centers = np.vstack((random_centers, np.random.multivariate_normal(means[1], cov, size=ncomp // 2)))
    #input_components = GaussianMixture([GaussianMixture.Component(mu, cov) for mu in random_centers]) --> does not work in python3
    input_components = GaussianMixture([GaussianMixture.Component(mu, hierarchical_cov) for mu in random_centers])
    initial_guess = GaussianMixture([GaussianMixture.Component(np.zeros(2) + 0.1, cov*3),
                     GaussianMixture.Component(np.zeros(2) - 0.1, cov*3),])

    def test_cluster(self):
        h = Hierarchical(self.input_components, self.initial_guess, verbose=True)
        sol = h.run()

        # both components should survive and have equal weight
        # expect precision loss in the weight summation for many input components
        self.assertEqual(len(sol.comp), 2)
        self.assertAlmostEqual(sol.w[0], 0.5, 13)

        # means should be reproduced
        eps = 12e-2
        self.assertAlmostEqual(sol.comp[0].mean[0], self.means[0][0], delta=eps)
        self.assertAlmostEqual(sol.comp[1].mean[0], self.means[1][0], delta=eps)

        # variance much larger now, but still should have little correlation
        for i in range(2):
            self.assertAlmostEqual(sol.comp[i].cov[0,1], 0, delta=0.1)

class TestKL(unittest.TestCase):
    def test_2D(self):
        d = 2
        mu1 = np.array([1, 3])
        mu2 = np.zeros(d)
        cov = np.eye(d)

        c1 = GaussianMixture.Component(mu1, cov)
        c2 = GaussianMixture.Component(mu2, cov, inv=True)

        self.assertAlmostEqual(kullback_leibler(c1, c2), 0.5 * mu1.dot(mu1), 15)

        # now add correlation, matrix inverse not trivial anymore
        cov2 = np.eye(d)
        cov2[0,1] = cov2[1,0] = 0.2
        c2 = GaussianMixture.Component(mu2, cov2, inv=True)
        self.assertAlmostEqual(kullback_leibler(c1, c2), 4.6045890027398722, 15)
