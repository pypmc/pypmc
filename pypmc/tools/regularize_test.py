'Unit tests for logsumexp 1D and 2D'

import numpy as np
from ._regularize import *

import unittest


class TestLogsumexp(unittest.TestCase):
    def test_1D(self):
        values  = np.array([1. ,2., 3. ])
        weights = np.array([ .3, .4, .3 ])
        target  = 2.28205254

        self.assertAlmostEqual(logsumexp(values, weights), target)

    def test_2D(self):
        values  = np.array([[4. ,8. , 3. ],
                            [ .3, .1, 5. ],
                            [2.3,5.6, 2.3]])
        weights = np.array([1.3, .4, .3 ])
        target  = np.array([7.14628895, 3.844190158, 4.82132340])

        np.testing.assert_allclose(logsumexp2D(values, weights), target)
