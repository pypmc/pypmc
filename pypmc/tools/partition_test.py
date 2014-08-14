'Unit tests for integer partition'

import numpy as np
from ._partition import *
from ..density.mixture import recover_gaussian_mixture

import unittest

class TestPartition(unittest.TestCase):
    def test_partition(self):
        N = 11
        k = 5
        target = [3, 2, 2, 2, 2]
        partition_result = partition(N, k)
        np.testing.assert_equal(partition_result, target)

class TestPatchData(unittest.TestCase):
    data = np.array([[ 0.16444512,  0.70628023],
                     [ 0.16444512,  0.69978308],
                     [ 0.16444512,  0.70075703],
                     [ 0.13286335,  0.72392304],
                     [ 0.53286335,  0.12392304],
                     [ 0.13286335,  0.72392304],
                     [ 0.53589102,  0.56556283],
                     [ 0.02999992,  0.86376941],
                     [ 0.37777816,  0.74790215],
                     [ 0.12766049,  0.97786075],
                     [ 0.95484293,  0.96036081],
                     [ 0.3541489 ,  0.16994458]])

    data_means = np.array([[ 0.16444512,  0.70227345],
                           [ 0.26619668,  0.52392304],
                           [ 0.31455637,  0.7257448 ],
                           [ 0.47888411,  0.70272205]])

    data_covs = np.array([[[ 0.00000000e+00,  0.00000000e+00],
                           [ 0.00000000e+00,  1.22778792e-05]],

                          [[ 0.053333333333, -0.08          ],
                           [-0.08          ,  0.12          ]],

                          [[ 0.066979197629, -0.036664392985],
                           [-0.036664392985,  0.022600002318]],

                          [[ 0.182726851097,  0.046223169791],
                           [ 0.046223169791,  0.212965433715]]])

    def test_patch_with_diag(self):
        patchmix = patch_data(self.data, 3, True)
        means, covs, weights = recover_gaussian_mixture(patchmix)

        # first covariance singular => skipped
        target_means = self.data_means[1:]
        # second covariance not positive definite => off-diagonals set to zero
        target_covs = np.vstack([ np.array([[[  self.data_covs[1][0,0],         0.              ],
                                             [          0.            , self.data_covs[1][1,1]  ]]]),

                                                             self.data_covs[2:]                         ])

        self.assertEqual(len(patchmix), 3)
        np.testing.assert_allclose(means  , target_means)
        np.testing.assert_allclose(covs   , target_covs )
        np.testing.assert_allclose(weights, 1./3.       )

    def test_patch_no_diag(self):
        patchmix = patch_data(self.data, 3, False)
        means, covs, weights = recover_gaussian_mixture(patchmix)

        self.assertEqual(len(patchmix), 2)
        np.testing.assert_allclose(means  , self.data_means[2:])
        np.testing.assert_allclose(covs   , self.data_covs [2:])
        np.testing.assert_allclose(weights, 1./2.              )
