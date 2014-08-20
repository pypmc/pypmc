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
    weights = [0.5, 0.5]
    components = [Gauss(m, c) for m,c in zip(means, covs)]
    input_components = MixtureDensity(components, weights)

    def setUp(self):
        import matplotlib.pyplot
        self.plt = matplotlib.pyplot

    def test_valid(self):
        self.plt.figure(figsize=(5,5))
        plot_mixture(self.input_components, 0, 1)
        # plt.savefig(self.__class__.__name__ + '.pdf')
        # saving a .pdf in python3 caused trouble --> .png is ok
        # self.plt.savefig(self.__class__.__name__ + '_python' + version[0] + '.png')

    def test_invalid(self):
        invalid_mix = MixtureDensity([Gauss(m, c) for m,c in zip(self.means, self.covs)],
                                       self.weights)

        # invalid covariance
        # must be hacked into MixtureProposal because it allows valid covariances only
        invalid_cov = np.array([[1, 2], [3, 4]])

        invalid_mix.components[0].sigma = invalid_cov

        expected_fail_args = ((self.input_components, -1, 1), # indices must be non negative
                              (self.input_components, 1, 1),  # indices must differ
                              (invalid_mix, 0, 1))            # covariance matrix is invalid

        for a in expected_fail_args:
            with self.assertRaises(AssertionError) as cm:
                plot_mixture(*a)
