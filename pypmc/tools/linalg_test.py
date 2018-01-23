'Unit tests for linear algebra'

import unittest
import numpy as np
from ._linalg import *

class TestLinalg(unittest.TestCase):
    def test_bilinear_sym(self):
        vector = np.array( [2. , 4.3, 7. ])
        matrix = np.array([[3. , 5. , 1.9],
                           [5. , .8 , 2.2],
                           [1.9, 2.2, 4.2]])
        target = 504.23200000000003
        evaluated = bilinear_sym(matrix, vector)
        self.assertAlmostEqual(evaluated, target)

    def test_chol_inv_det(self):
        matrices = [np.array([[1.0, 0.5],
                              [0.5, 1.0]]),
                    np.array([[1.0, 0.5, 0.1],
                              [0.5, 2.1,-0.4],
                              [0.1,-0.4, 1.8]])]
        for matrix in matrices:
            matrix_copy = matrix.copy()
            l, inv, log_det = chol_inv_det(matrix)

            # input unchanged
            np.testing.assert_allclose(matrix, matrix_copy)
            # compare against numpy
            np.testing.assert_allclose(l, np.linalg.cholesky(matrix))
            np.testing.assert_allclose(inv, np.linalg.inv(matrix))
            self.assertAlmostEqual(log_det, np.log(np.linalg.det(matrix)))

        asymmetric_sigma  = np.array([[0.01 , 0.003 ]
                             ,[0.001, 0.0025]])
        self.assertRaisesRegexp(np.linalg.LinAlgError, 'not symmetric',
                                chol_inv_det, asymmetric_sigma)

        singular_sigma = np.array([[0.0, 0.0   , 0.0]
                                   ,[0.0, 0.0025, 0.0]
                                   ,[0.0, 0.0   , 0.6]])
        negative_sigma = -1.0 * np.eye(13)

        for matrix in [singular_sigma, negative_sigma]:
            self.assertRaisesRegexp(np.linalg.LinAlgError, 'not positive definite',
                                    chol_inv_det, matrix)
