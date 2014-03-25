"""Unit tests for the R-value clustering.

"""
from .r_value import *
import numpy as np
import unittest

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
