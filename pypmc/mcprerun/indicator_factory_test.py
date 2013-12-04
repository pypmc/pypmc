"""Unit tests for the indicator factory.

"""

from .indicator_factory import *
import numpy as np
import unittest

class TestBall(unittest.TestCase):
    def test_unit_ball(self):
        unit_ball_ind = ball()

        inside  = [np.array((0.,0.,0.)) , np.array((0.,0.,1. )) , np.array(( .5,.5,.5))]
        outside = [np.array((1.,1.,1.)) , np.array((0.,0.,1.1)) , np.array((5  ,.5,.5))]

        for point in inside:
            self.assertTrue (unit_ball_ind(point))
        for point in outside:
            self.assertFalse(unit_ball_ind(point))

    def test_ball_boundary(self):
        unit_ball_with_bdy    = ball(bdy = True )
        unit_ball_without_bdy = ball(bdy = False)

        bdy_point = np.array((0.,0.,1.))

        self.assertTrue (unit_ball_with_bdy   (bdy_point))
        self.assertFalse(unit_ball_without_bdy(bdy_point))

    def test_ball_wrong_dimension(self):
        unit_ball_ind = ball() # unit ball in dim 3
        point3d       = np.array((0.,0.,1.))
        point2d       = np.array((0.,0.))

        unit_ball_ind(point3d)
        self.assertRaisesRegexp(ValueError, 'input has wrong dimension', unit_ball_ind, point2d)

class TestHyperrectangle(unittest.TestCase):
    def test_unit_hr(self):
        unit_hr_ind = hyperrectangle()

        inside  = [np.array(( .1, .1, .1)) , np.array(( .1, .1,1. )) , np.array(( .5,.5,.5))]
        outside = [np.array((1.1,1.1,1.5)) , np.array((0.1,0.1,1.1)) , np.array((5  ,.5,.5))]

        for point in inside:
            self.assertTrue (unit_hr_ind(point))
        for point in outside:
            self.assertFalse(unit_hr_ind(point))

    def test_hr_boundary(self):
        unit_hr_with_bdy    = hyperrectangle(bdy = True )
        unit_hr_without_bdy = hyperrectangle(bdy = False)

        bdy_points = [np.array((0.,0.,0.)) , np.array((0.,0.,1.)) , np.array((0.,1.,0.)) ,
                      np.array((0.,1.,1.)) , np.array((1.,0.,0.)) , np.array((1.,0.,1.)) ,
                      np.array((1.,1.,0.)) , np.array((1.,1.,1.)) ,

                      np.array((.5,1.,.3)) , np.array((0.,.6,1.)) , np.array((.4,.7,0.))]

        for point in bdy_points:
            self.assertTrue (unit_hr_with_bdy   (point))
            self.assertFalse(unit_hr_without_bdy(point))

    def test_hr_wrong_dimension(self):
        unit_hr_ind = hyperrectangle() # unit hyperrectangle in dim 3 ([0,1]**3)
        point3d     = np.array((0.,0.,1.))
        point2d     = np.array((0.,0.))

        unit_hr_ind(point3d)
        self.assertRaisesRegexp(ValueError, 'input has wrong dimension', unit_hr_ind, point2d)

    def test_hr_wrong_init(self):
        # hyperrectangle should raise an error if any lower > upper
        self.assertRaises(ValueError, hyperrectangle, ((1.,1.),(10.,0.)))

        # hyperrectangle should raise an error if dim(lower)!=dim(upper)
        self.assertRaises(ValueError, hyperrectangle, ((1.,1.,1.),(10.,10.)))
