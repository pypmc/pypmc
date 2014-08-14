"""Unit tests for the R-value clustering.

"""
from .r_value import *
from ..density.mixture import recover_gaussian_mixture, recover_t_mixture
import numpy as np
import unittest

# output from ten two-dimensional Markov chains after n steps
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

        indices_3_dim = (0,2)

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
        self.assertRaisesRegexp(AssertionError, 'All.*indices.*less than 2',
                                multivariate_r, multivariate_means, multivariate_covs, 10, indices=indices_3_dim)

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

    def test_indices(self):
        calculated_multivariate_r = multivariate_r(multivariate_means, multivariate_covs, n, indices=(1,1))
        calculated_multivariate_approx_r = multivariate_r(multivariate_means, multivariate_covs, n, approx=True, indices=(0,))

        target_multivariate_r        = [135.6615268882104   , 135.6615268882104]
        target_multivariate_approx_r = [ 64.553881518372663]


        self.assertEqual(len(calculated_multivariate_approx_r), 1)
        self.assertEqual(len(calculated_multivariate_r)       , 2)

        self.assertAlmostEqual(calculated_multivariate_approx_r[0], target_multivariate_approx_r[0])
        self.assertAlmostEqual(calculated_multivariate_r       [0], target_multivariate_r       [0])
        self.assertAlmostEqual(calculated_multivariate_r       [1], target_multivariate_r       [1])

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

    def test_indices(self):
        means3d = np.array ([[  4.31915681    ,  1.08038315     ,   0.1           ],
                             [  4.31864843    ,  1.11007763     ,  -0.2           ],
                             [  4.31389518    ,  1.10076274     ,   0.            ],
                             [  4.29077485    ,  1.1090809      ,   0.08          ],
                             [  4.29175174    ,  1.09263979     ,  -0.8           ]])
        covs_3d = np.array([[[  9.27501935e-03,   2.93898176e-03,   2.94712784e-03],
                             [  2.93898176e-03,   2.27907650e-03,   2.18398826e-03],
                             [  2.94712784e-03,   2.18398826e-03,   5.71229368e-02]],

                            [[  9.14057218e-03,   2.45045703e-03,  -6.75693542e-04],
                             [  2.45045703e-03,   2.16242986e-03,  -2.27529109e-04],
                             [ -6.75693542e-04,  -2.27529109e-04,   4.42518545e-02]],

                            [[  1.20508676e-02,   4.44680466e-03,  -1.47851744e-04],
                             [  4.44680466e-03,   3.04590236e-03,  -6.84309488e-05],
                             [ -1.47851744e-04,  -6.84309488e-05,   3.66323621e-02]],

                            [[  1.05334136e-02,   3.89969309e-03,  -2.94526373e-03],
                             [  3.89969309e-03,   2.75030883e-03,  -8.85690874e-04],
                             [ -2.94526373e-03,  -8.85690874e-04,   3.17552111e-02]],

                            [[  7.41327996e-03,   2.53320165e-03,  -1.90328907e-03],
                             [  2.53320165e-03,   2.09785528e-03,  -1.92942666e-04],
                             [ -1.90328907e-03,  -1.92942666e-04,   5.02877762e-02]]])
        n = 500

        # target_r_values        = [1.02977183,  1.07893712,  7.1196891 ]
        # target_approx_r_values = [1.01934397,  1.06018193,  4.18999978]

        # grouping using all dimensions should result in at least two groups
        inferred_groups_all_dim        = r_group(means3d, covs_3d, n, approx=False)
        inferred_groups_all_dim_approx = r_group(means3d, covs_3d, n, approx=True)
        self.assertGreaterEqual(len(inferred_groups_all_dim_approx), 2)
        self.assertGreaterEqual(len(inferred_groups_all_dim       ), 2)

        # neglecting the last dimension, the result should be one group
        # with standard critical_r = 1.5
        target_group_partial = range(5)
        inferred_groups_partial_dim        = r_group(means3d, covs_3d, n, indices=(0,1), approx=False)
        inferred_groups_partial_dim_approx = r_group(means3d, covs_3d, n, indices=(0,1), approx=True)
        self.assertEqual(len(inferred_groups_partial_dim_approx   ), 1)
        self.assertEqual(len(inferred_groups_partial_dim          ), 1)
        self.assertEqual(len(inferred_groups_partial_dim_approx[0]), 5)
        self.assertEqual(len(inferred_groups_partial_dim       [0]), 5)
        for i in range(5):
            self.assertEqual(inferred_groups_partial_dim_approx[0][i], target_group_partial[i])
            self.assertEqual(inferred_groups_partial_dim       [0][i], target_group_partial[i])

# making mixtures out of data:

# ``n`` must be equal for all chains
wrong_data = (
# chain 0
[[ 0.0,  1.0],
 [ 0.1, -0.1],
 [-0.1,  0.1]],
# chain 1
[[ 0.5,  0.5],
 [-0.3,  0.0],
 [-0.2,  0.0],
 [ 0.0,  0.0]]
)

data = (
# chain 0
[[ 0.0,  1.0],
 [ 1.0,  0.0],
 [-1.0, -1.0],
 [ 0.0,  0.0],
 [ 0.1, -0.1],
 [-0.1,  0.1]],
# chain 1
[[ 0.5,  0.5],
 [-0.3,  0.0],
 [ 0.1, -0.4],
 [-0.1, -0.1],
 [ 0.1,  0.1],
 [-0.2,  0.0]],
# chain 2
[[ 1.0, 1.2],
 [ 1.4, 1.5],
 [ 1.2, 1.3],
 [ 1.3, 1.8],
 [ 1.9, 1.4],
 [ 1.5, 1.8]],
 # chain 3
[[-0.1, -0.5],
 [ 0.3,  0.0],
 [-0.5,  0.4],
 [ 0.1,  0.1],
 [-0.1,  0.1],
 [-0.2,  0.0]]
)

critical_r = 2.

target_groups = [[0,1,3], [2]]
groups = r_group([np.mean(dat, axis=0) for dat in data], [np.cov(dat, rowvar=0) for dat in data], 6)
np.testing.assert_equal(groups, target_groups)

target_means = np.array([
                         # chains 0, 1, 3 --> group 0
                         [ 0.033333333333,  0.011111111111], [-0.077777777778,  0.011111111111],
                         # chain 2        --> group 1
                         [1.2, 1.333333333], [1.566666667, 1.666666667]
                       ])

target_covs  = np.array([
                         # chains 0, 1, 3 --> group 0
                         [[ 0.295       ,  0.1483333333],
                          [ 0.1483333333,  0.3036111111]],

                         [[ 0.0519444444, -0.0152777778],
                          [-0.0152777778,  0.0561111111]],

                         # chain 2        --> group 1
                         [[ 0.04        ,  0.03        ],
                          [ 0.03        ,  0.0233333333]],

                         [[ 0.0933333333, -0.0666666667],
                          [-0.0666666667,  0.0533333333]]
                       ])

target_weights = np.ones(4) / 4. # equal weights

class TestMakeRGaussmix(unittest.TestCase):
    def test_error_messages(self):
        self.assertRaisesRegexp(AssertionError, 'Every chain.*same.*number.*points',
                                make_r_gaussmix, wrong_data)

    def test_make_r_gaussmix(self):
        inferred_mixture = make_r_gaussmix(data, K_g=2, critical_r=critical_r)
        inferred_means, inferred_covs, inferred_weights = recover_gaussian_mixture(inferred_mixture)

        np.testing.assert_allclose(inferred_means  , target_means  )
        np.testing.assert_allclose(inferred_covs   , target_covs   )
        np.testing.assert_allclose(inferred_weights, target_weights)

class TestMakeRTmix(unittest.TestCase):
    valid_dof = 5.
    invalid_dof = 1. # for finite covariance: dof > 2

    def test_error_messages(self):
        self.assertRaisesRegexp(AssertionError, 'Every chain.*same.*number.*points',
                                make_r_tmix, wrong_data)
        self.assertRaisesRegexp(AssertionError, 'dof.*(greater|larger).*?2',
                                make_r_tmix, data, dof=self.invalid_dof)
        self.assertRaisesRegexp(AssertionError, 'dof.*(greater|larger).*?2',
                                make_r_tmix, data, dof=2)

    def test_make_r_tmix(self):
        target_dofs = np.array([self.valid_dof] * 4)

        inferred_mixture = make_r_tmix(data, K_g=2, critical_r=critical_r, dof=self.valid_dof)
        inferred_means, inferred_covs, inferred_dofs, inferred_weights = recover_t_mixture(inferred_mixture)

        np.testing.assert_allclose(inferred_means                                          , target_means  )
        np.testing.assert_allclose(inferred_covs * self.valid_dof / (self.valid_dof - 2.)  , target_covs   )
        np.testing.assert_allclose(inferred_weights                                        , target_weights)
        np.testing.assert_allclose(inferred_dofs                                           , target_dofs   )
