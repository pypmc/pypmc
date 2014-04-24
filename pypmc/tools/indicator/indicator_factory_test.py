"""Unit tests for the indicator factory.

"""

from . import *
import numpy as np
import unittest

class TestBall(unittest.TestCase):
    # unit ball in dim 3
    center = [0., 0., 0.]

    def test_unit_ball(self):
        unit_ball_ind = ball(self.center)

        inside  = [np.array((0.,0.,0.)) , np.array((0.,0.,1. )) , np.array(( .5,.5,.5))]
        outside = [np.array((1.,1.,1.)) , np.array((0.,0.,1.1)) , np.array((5. ,.5,.5))]

        for point in inside:
            self.assertTrue (unit_ball_ind(point))
        for point in outside:
            self.assertFalse(unit_ball_ind(point))

    def test_ball_boundary(self):
        unit_ball_with_bdy    = ball(self.center, bdy=True)
        unit_ball_without_bdy = ball(self.center, bdy=False)

        bdy_point = np.array((0.,0.,1. ))
        inside    = np.array((0.,0., .5))
        outside   = np.array((0.,0.,1.5))

        self.assertTrue (unit_ball_with_bdy   (inside   ))
        self.assertTrue (unit_ball_without_bdy(inside   ))

        self.assertFalse(unit_ball_with_bdy   (outside  ))
        self.assertFalse(unit_ball_without_bdy(outside  ))

        self.assertTrue (unit_ball_with_bdy   (bdy_point))
        self.assertFalse(unit_ball_without_bdy(bdy_point))

    def test_ball_wrong_dimension(self):
        # unit ball in dim 3
        unit_ball_with_bdy = ball(self.center, bdy=True)
        unit_ball_no_bdy   = ball(self.center, bdy=False)

        point3d       = np.array((.4,.1,.2))
        point2d       = np.array((0.,0.))

        self.assertTrue(unit_ball_with_bdy(point3d))
        self.assertTrue(unit_ball_no_bdy(point3d))
        self.assertRaisesRegexp(ValueError, 'input has wrong dimension', unit_ball_no_bdy  , point2d)
        self.assertRaisesRegexp(ValueError, 'input has wrong dimension', unit_ball_with_bdy, point2d)

    def test_input_copy(self):
        center_array = np.array([.1, 2.])
        radius = .2
        ball_indicator = ball(center_array, radius)

        self.assertTrue(  ball_indicator( (0., 2.) )  )

        # change center_array
        center_array[0] = 100.

        # change in ``center_array`` should not have changed ``ball_indicator``
        self.assertTrue(  ball_indicator( (0., 2.) )  )

        second_ball = ball(center_array, radius)

        self.assertFalse(  second_ball( (0., 2.) )  )

class TestHyperrectangle(unittest.TestCase):
    lower = np.array([0, 0, 0.])
    upper = np.array([1, 1, 1.])

    def test_unit_hr(self):
        unit_hr_with_bdy = hyperrectangle(self.lower, self.upper, bdy=True)
        unit_hr_no_bdy   = hyperrectangle(self.lower, self.upper, bdy=False)

        inside  = [( .1, .1, .1) , np.array(( .1, .1, .9)) , np.array(( .5,.5,.5))]
        outside = [(1.1,1.1,1.5) , np.array((0.1,0.1,1.1)) , np.array((5. ,.5,.5))]

        for point in inside:
            self.assertTrue (unit_hr_with_bdy(point))
            self.assertTrue (unit_hr_no_bdy(point))
        for point in outside:
            self.assertFalse(unit_hr_with_bdy(point))
            self.assertFalse(unit_hr_no_bdy(point))

    def test_hr_boundary(self):
        unit_hr_with_bdy    = hyperrectangle(self.lower, self.upper, bdy = True )
        unit_hr_without_bdy = hyperrectangle(self.lower, self.upper, bdy = False)

        bdy_points = [np.array((0.,0.,0.)) , np.array((0.,0.,1.)) , np.array((0.,1.,0.)) ,
                      np.array((0.,1.,1.)) , np.array((1.,0.,0.)) , np.array((1.,0.,1.)) ,
                      np.array((1.,1.,0.)) , np.array((1.,1.,1.)) ,

                      np.array((.5,1.,.3)) , np.array((0.,.6,1.)) , np.array((.4,.7,0.))]

        for point in bdy_points:
            self.assertTrue (unit_hr_with_bdy   (point))
            self.assertFalse(unit_hr_without_bdy(point))

    def test_hr_wrong_dimension(self):
        # unit hyperrectangle in dim 3 ([0,1]**3)
        unit_hr_with_bdy = hyperrectangle(self.lower, self.upper, bdy=True)
        unit_hr_no_bdy   = hyperrectangle(self.lower, self.upper, bdy=False)

        point3d     = np.array((.1,.1,.1))
        point2d     = np.array((.1,.1))

        self.assertTrue(unit_hr_with_bdy(point3d))
        self.assertTrue(unit_hr_no_bdy(point3d))
        self.assertRaisesRegexp(ValueError, 'input has wrong dimension', unit_hr_with_bdy, point2d)
        self.assertRaisesRegexp(ValueError, 'input has wrong dimension', unit_hr_no_bdy  , point2d)

    def test_hr_wrong_init(self):
        # hyperrectangle should raise an error if any lower > upper
        self.assertRaises(ValueError, lambda: hyperrectangle(np.array([1.,1.]), np.array([10.,0.]), bdy=True))
        self.assertRaises(ValueError, lambda: hyperrectangle(np.array([1.,1.]), np.array([10.,0.]), bdy=False))

        # hyperrectangle should raise an error if dim(lower)!=dim(upper)
        self.assertRaises(ValueError, lambda: hyperrectangle(np.array([1.,1.,1.]), np.array([10.,10.]), bdy=True))
        self.assertRaises(ValueError, lambda: hyperrectangle(np.array([1.,1.,1.]), np.array([10.,10.]), bdy=False))

    def test_input_copy(self):
        lower = (.1, 2.)
        upper = [1., 5.]
        hr_indicator = hyperrectangle(lower, upper)

        self.assertTrue(  hr_indicator( (0.5, 2.5) )  )

        # change upper
        upper[1] = 2.1

        # change in ``upper`` should not have changed ``hr_indicator``
        self.assertTrue(  hr_indicator( (0.5, 2.5) )  )

        second_hr = hyperrectangle(lower, upper)

        self.assertFalse(  second_hr( (0.5, 2.5) )  )
