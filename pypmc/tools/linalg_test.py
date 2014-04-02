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
