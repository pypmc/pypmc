from sys import version
import numpy as np
from ._plot import *
from ..density.gauss import Gauss
from ..density.mixture import MixtureDensity

import unittest

def setUpModule():
    try:
        import matplotlib
    except ImportError:
        raise unittest.SkipTest("Can't test plot without matplotlib")

class TestPlotMixture(unittest.TestCase):
    means  = (np.array([ -1,  -1]),
              np.array([ 1  , 1]))

    covs = [np.array([[1, -0.2], [-0.2, 1.]]), np.array([[10., -0.2], [-0.2, 0.9]])]
    equal_weights = [0.5, 0.5]
    unequal_weights = [0.3, 0.7]
    components = [Gauss(m, c) for m,c in zip(means, covs)]
    input_components_equal_weight = MixtureDensity(components, equal_weights)
    input_components_unequal_weight = MixtureDensity(components, unequal_weights)

    def setUp(self):
        import matplotlib.pyplot
        self.plt = matplotlib.pyplot

    def test_valid(self):
        self.plt.figure(figsize=(5,5))
        self.plt.subplot(221)
        plot_mixture(self.input_components_unequal_weight, 0, 1, visualize_weights=True)
        self.plt.colorbar()
        self.plt.subplot(222)
        plot_mixture(self.input_components_equal_weight, 0, 1)
        self.plt.subplot(223)
        plot_mixture(self.input_components_equal_weight, 0, 1, visualize_weights=True)
        # plt.savefig(self.__class__.__name__ + '.pdf')
        # saving a .pdf in python3 caused trouble --> .png is ok
        # self.plt.savefig(self.__class__.__name__ + '_python' + version[0] + '.png')

    def test_invalid(self):
        invalid_mix = MixtureDensity([Gauss(m, c) for m,c in zip(self.means, self.covs)],
                                       self.equal_weights)

        # invalid covariance
        # must be hacked into MixtureProposal because it allows valid covariances only
        invalid_cov = np.array([[1, 2], [3, 4]])

        invalid_mix.components[0].sigma = invalid_cov

        expected_fail_args = ((self.input_components_equal_weight, -1, 1), # indices must be non negative
                              (self.input_components_equal_weight, 1, 1),  # indices must differ
                              (invalid_mix, 0, 1))                         # covariance matrix is invalid

        for a in expected_fail_args:
            with self.assertRaises(AssertionError) as cm:
                plot_mixture(*a)

class TestPlotResponsibility(unittest.TestCase):
    def setUp(self):
        import matplotlib.pyplot
        self.plt = matplotlib.pyplot

        np.random.seed(12531235)

        self.N = 100
        self.D = 2
        self.K = 10
        self.data = np.random.rand(self.N, self.D)
        self.r = np.random.rand(self.N, self.K)

    def test_valid(self):
        self.plt.figure()
        plot_responsibility(self.data, self.r)
        # saving a .pdf in python3 caused trouble --> .png is ok
        # self.plt.savefig(self.__class__.__name__ + '_python' + version[0] + '.png')

