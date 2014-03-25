"""Unit tests for the R-value clustering.

"""
from .r_value import *
import numpy as np
import unittest

# output from ten two-dimensional markov-chains after n steps
n = 10000

multivariate_means = np.array([[  9.43294669,  -9.96858978],
                               [-10.63298025, -10.036594  ],
                               [  9.3900033 ,  10.02509706],
                               [-10.53628855, -10.02974689],
                               [  9.46250514,  -9.97144287],
                               [-10.61542827,   9.97603709],
                               [  9.42639861, -10.03105987],
                               [-10.60701999, -10.00775056],
                               [-10.56959015, -10.04381837],
                               [-10.54281051,  10.01823405]])

multivariate_covs  = np.array([[[  1.62468690e+00,  -8.33968893e-03],
                                [ -8.33968893e-03,   9.43557917e-01]],

                               [[  1.72159595e+00,   2.85677262e-02],
                                [  2.85677262e-02,   9.55759350e-01]],

                               [[  1.69905429e+00,  -1.22852589e-02],
                                [ -1.22852589e-02,   9.82452274e-01]],

                               [[  1.50562855e+00,   2.02680124e-02],
                                [  2.02680124e-02,   9.83057545e-01]],

                               [[  1.54786354e+00,  -3.38487390e-02],
                                [ -3.38487390e-02,   1.07495261e+00]],

                               [[  1.81265715e+00,  -7.90566591e-04],
                                [ -7.90566591e-04,   9.54108371e-01]],

                               [[  1.76233959e+00,  -3.22287920e-02],
                                [ -3.22287920e-02,   9.78554160e-01]],

                               [[  1.79721672e+00,   5.05481975e-02],
                                [  5.05481975e-02,   9.72505455e-01]],

                               [[  1.62568169e+00,  -1.14017726e-03],
                                [ -1.14017726e-03,   9.65515639e-01]],

                               [[  1.70721411e+00,   3.91543994e-02],
                                [  3.91543994e-02,   9.78570335e-01]]])

class TestRValue(unittest.TestCase):
    def test_ivalid_input(self):
        # ``means`` and ``variances`` must be vector-like and equally long

        means = np.array( [1., 2., 3.] )
        means_matrix_like = np.array( [[1., 2.], [.5, 2.1]] )

        variances = np.array( [1., 2., 3.] )
        variances_matrix_like = np.array( [[[1.00, 0.01],
                                            [0.01, 4.00]],

                                           [[0.01, 0.00],
                                            [0.00, 0.04]]] )
        variances_too_few = np.array( [1., 2.] )

        self.assertRaises(AssertionError, r_value, means, variances_matrix_like, 10)
        self.assertRaises(AssertionError, r_value, means_matrix_like, variances, 10)
        self.assertRaises(AssertionError, r_value, means, variances_too_few, 10)

    def test_two_components(self):
        means     = np.array( (0.229459   , 0.318064   ) )
        variances = np.array( (0.000443577, 0.000147002) )
        n = 2000

        calculated_R        = r_value(means, variances, n)
        calculated_approx_R = r_value(means, variances, n, approx=True)

        target_R        = + np.inf
        target_approx_R = 14.292973057795828

        self.assertAlmostEqual(calculated_approx_R, target_approx_R)
        np.testing.assert_equal(calculated_R, target_R)

    def test_four_components(self):
        means     = np.array( (2.851685e+01, 2.851675e+01, 2.851946e+01, 2.851998e+01) )
        variances = np.array( (2.419658e-02, 2.355381e-02, 2.526591e-02, 2.432331e-02) )
        n = 2354

        calculated_R        = r_value(means, variances, n)
        calculated_approx_R = r_value(means, variances, n, approx=True)

        target_R        = 0.99993379654073156
        target_approx_R = 0.999693904056566

        self.assertAlmostEqual(calculated_approx_R, target_approx_R)
        self.assertAlmostEqual(calculated_R, target_R)

    def test_five_components(self):
        means     = np.array( (2.3585e+01  , 2.451675e+01, 2.951946e+01, 2.851998e+01, 2.512137e+01) )
        variances = np.array( (2.419658e-02, 2.355381e-02, 2.526591e-02, 2.432331e-02, 2.145637e-02) )
        n = 9238

        calculated_R        = r_value(means, variances, n)
        calculated_approx_R = r_value(means, variances, n, approx=True)

        target_R        = 685.53010544500285
        target_approx_R = 287.43985146937433

        self.assertAlmostEqual(calculated_approx_R, target_approx_R)
        self.assertAlmostEqual(calculated_R, target_R)

class TestMultivariateR(unittest.TestCase):
    def test_invalid_input(self):
        # ``means`` and ``covs`` must be equally long
        two_means = np.array( [[1., 2.], [.5, 2.1]] )

        three_covariances = np.array( [[[1.00, 0.01],
                                        [0.01, 4.00]],

                                       [[0.01, 0.00],
                                        [0.00, 0.04]],

                                       [[0.50, 0.00],
                                        [0.00, 0.03]]] )

        two_covs_wrong_dimension = np.array(  [[[1.00, 0.01, 0.00],
                                                [0.01, 4.00, 0.00],
                                                [0.00, 0.00, 2.00]],

                                               [[0.01, 0.00, 0.03],
                                                [0.00, 0.10, 0.00],
                                                [0.03, 0.00, 0.75]]]   )

        two_covs_not_square_matrices = np.array( [[[1.00, 0.01],
                                                   [0.01, 4.00],
                                                   [0.00, 0.00]],

                                                  [[0.01, 0.00],
                                                   [0.00, 0.10],
                                                   [0.03, 0.00]]]   )

        # ``means`` must be 2d
        three_means_wrong_shape = np.array(  [1., 2., 3.]  )

        # ``covs`` must be 3d
        two_covariances_wrong_shape = np.array(  [1., 2.]  )

        self.assertRaisesRegexp(AssertionError, '.*means.*not match.*covs',
                                multivariate_r, two_means, three_covariances, 10)
        self.assertRaisesRegexp(AssertionError, '.*means.*must.*[Mm]atrix',
                                multivariate_r, three_means_wrong_shape, three_covariances, 10)
        self.assertRaisesRegexp(AssertionError, '.*covs.*must.*3[\ -]?[Dd]im',
                                multivariate_r, two_means, two_covariances_wrong_shape, 10)
        self.assertRaisesRegexp(AssertionError, '.*covs\.shape\[1\].*must match .*covs\.shape\[2\]',
                                multivariate_r, two_means, two_covs_not_square_matrices, 10)
        self.assertRaisesRegexp(AssertionError, 'Dimensionality.*means.*covs.*not match',
                                multivariate_r, two_means, two_covs_wrong_dimension, 10)


    def test_multivariate_r(self):
        calculated_multivariate_r = multivariate_r(multivariate_means, multivariate_covs, n)
        calculated_multivariate_approx_r = multivariate_r(multivariate_means, multivariate_covs, n, approx=True)

        target_multivariate_r        = [90.444063973857212, 135.6615268882104  ]
        target_multivariate_approx_r = [64.553881518372663,  96.528380573770065]


        self.assertEqual(len(calculated_multivariate_approx_r), 2)
        self.assertEqual(len(calculated_multivariate_r)       , 2)

        for i in range(2):
            self.assertAlmostEqual(calculated_multivariate_approx_r[i], target_multivariate_approx_r[i])
            self.assertAlmostEqual(calculated_multivariate_r       [i], target_multivariate_r       [i])

class TestRGroup(unittest.TestCase):
    def test_group(self):
        target_groups = [[0, 4, 6], [1, 3, 7, 8], [2], [5, 9]]

        inferred_groups        = r_group(multivariate_means, multivariate_covs, n)
        inferred_groups_approx = r_group(multivariate_means, multivariate_covs, n, approx=True)

        np.testing.assert_equal(inferred_groups,        target_groups)
        np.testing.assert_equal(inferred_groups_approx, target_groups)

    def test_critical_r(self):
        # this R value should not group any components together
        critical_r = 1.0

        target_groups = [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]]

        inferred_groups        = r_group(multivariate_means, multivariate_covs, n, critical_r)
        inferred_groups_approx = r_group(multivariate_means, multivariate_covs, n, critical_r, approx=True)

        np.testing.assert_equal(inferred_groups,        target_groups)
        np.testing.assert_equal(inferred_groups_approx, target_groups)
