"""Unit tests for the StudentT probability densities

"""

from .student_t import *
import numpy as np
import unittest

rng_seed  = 12850419274 % 4294967296
rng_steps = 50000

singular_sigma = np.array([[0.0, 0.0   , 0.0]
                          ,[0.0, 0.0025, 0.0]
                          ,[0.0, 0.0   , 0.6]])

asymmetric_sigma  = np.array([[0.01 , 0.003 ]
                             ,[0.001, 0.0025]])

offdiag_sigma  = np.array([[0.01 , 0.003 ]
                          ,[0.003, 0.0025]])

class FakeRNG(object):
    def normal(self, a, b, N):
        return np.array(N*[1.])
    def chisquare(self, degree_of_freedom):
        # print 'in FakeRNG: degree_of_freedom', degree_of_freedom
        assert type(degree_of_freedom) == float
        return degree_of_freedom
fake_rng = FakeRNG()

class TestLocalStudentT(unittest.TestCase):
    def setUp(self):
        print('"StudentT" needs .LocalGauss.')
        print('When this test fails, first make sure that .LocalGauss works.')

    def test_update(self):
        dof = 1.5
        t = LocalStudentT(offdiag_sigma, dof)
        sample = t.propose(np.array([0.,0.]), fake_rng)
        evaluate_at_sample = t.evaluate(np.array([0.,0.]), sample)

        self.assertRaises(np.linalg.LinAlgError, t.update, singular_sigma)
        self.assertRaises(np.linalg.LinAlgError, t.update, asymmetric_sigma)

        # check that the internal variables of ``g`` do not change after LinAlgError
        np.testing.assert_equal(t.sigma, offdiag_sigma)
        self.assertEqual(t.dim, 2)
        np.testing.assert_equal(t.propose (np.array([0.,0.]), fake_rng), sample)
        np.testing.assert_equal(t.evaluate(np.array([0.,0.]), sample), evaluate_at_sample)

    def test_bad_dof(self):
        expected_error_msg = ".*dof.*must.*((larger)|(greater)).*(0|(zero))"
        self.assertRaisesRegexp(AssertionError, expected_error_msg, LocalStudentT, offdiag_sigma,  0.0)
        self.assertRaisesRegexp(AssertionError, expected_error_msg, LocalStudentT, offdiag_sigma, -1.1)

    def test_badCovarianceInput(self):
        self.assertRaises(np.linalg.LinAlgError, lambda: LocalStudentT(singular_sigma, 10) )
        self.assertRaises(np.linalg.LinAlgError, lambda: LocalStudentT(asymmetric_sigma,10) )

    def test_evaluate(self):
        sigma = np.array([[0.0049, 0.  ]
                         ,[0.    ,  .01]])
        delta = 1e-9
        dof   = 5.

        t = LocalStudentT(sigma=sigma, dof=dof)

        x0 = np.array([1.25, 4.3  ])
        x1 = np.array([1.3 , 4.4  ])
        x2 = np.array([1.26, 4.424])

        target1 = 2.200202941
        target2 = 2.174596526

        self.assertAlmostEqual(t.evaluate(x1, x0), target1, delta=delta)
        self.assertAlmostEqual(t.evaluate(x0, x1), target1, delta=delta)

        self.assertAlmostEqual(t.evaluate(x2, x0), target2, delta=delta)
        self.assertAlmostEqual(t.evaluate(x0, x2), target2, delta=delta)

    def test_propose_covariance(self):
        sigma = np.array([[.2]])
        dof   = 5.
        delta = 0.005

        t = LocalStudentT(sigma = sigma, dof = dof)

        np.random.seed(rng_seed)
        current = np.array([0.])
        target_mean = current[0]
        target_var  = dof / (dof - 2.) * sigma[0,0]

        mean = 0.
        var  = 0.

        for i in range(rng_steps-1):
            proposal = t.propose(current, np.random)
            mean += proposal[0]
            var  += proposal[0]**2

        # test if value for rng can be omitted
        proposal = t.propose(current)
        mean += proposal[0]
        var  += proposal[0]**2

        mean /= rng_steps
        var  /= rng_steps
        var  -= mean**2

        self.assertAlmostEqual(mean, target_mean, delta=delta)
        self.assertAlmostEqual(var , target_var , delta=delta)

    def test_propose_2D(self):
        sigma      = offdiag_sigma
        dof        = 5.
        delta_mean = .001
        delta_var0 = .0006
        delta_var1 = .00004

        t = LocalStudentT(sigma=sigma, dof=dof)

        np.random.seed(rng_seed)
        current = np.array([4.3, 1.1])
        target_mean0 = current[0]
        target_var0  = offdiag_sigma[0,0] * dof/(dof-2)
        target_mean1 = current[1]
        target_var1  = offdiag_sigma[1,1] * dof/(dof-2)

        values0 = []
        values1 = []

        for i in range(rng_steps-1):
            proposal = t.propose(current, np.random)
            values0 += [proposal[0]]
            values1 += [proposal[1]]

        # test if value for rng can be omitted
        proposal = t.propose(current)
        values0 += [proposal[0]]
        values1 += [proposal[1]]

        values0 = np.array(values0)
        values1 = np.array(values1)

        mean0 = values0.mean()
        var0  = values0.var()
        mean1 = values1.mean()
        var1  = values1.var()

        self.assertAlmostEqual(mean0, target_mean0, delta=delta_mean)
        self.assertAlmostEqual(var0 , target_var0 , delta=delta_var0)
        self.assertAlmostEqual(mean1, target_mean1, delta=delta_mean)
        self.assertAlmostEqual(var1 , target_var1 , delta=delta_var1)

    def test_1D(self):
        sigma = 1
        dof = 1 # Cauchy
        x = np.array([3.2])
        y = np.array([0.])
        current = np.empty(1)

        # pass number as scalar, vector, array
        for s in (sigma, [sigma], [[sigma]]):
            t = LocalStudentT(sigma=s, dof=dof)
            self.assertAlmostEqual(t.evaluate(x, y), -3.5642087303149452)
            t.propose(current, np.random)

class TestStudentT(unittest.TestCase):
    mean  = np.array( [1.25, 4.3   ] )
    sigma = np.array([[0.0049, 0.  ]
                     ,[0.    ,  .01]])
    dof   = 5.

    t = StudentT(mean, sigma, dof)

    point1 = np.array([1.3 , 4.4  ])
    point2 = np.array([1.26, 4.424])

    target1 = 2.200202941
    target2 = 2.174596526

    delta = 1e-9

    def setUp(self):
        print('"StudentT" needs .LocalStudentT.')
        print('When this test fails, first make sure that .LocalStudentT works.')

    def test_update(self):
        dof = 1.5

        t = StudentT(self.mean, offdiag_sigma, dof)
        sample = t.propose(1, fake_rng)[0]
        evaluate_at_sample = t.evaluate(sample)

        self.assertRaises(np.linalg.LinAlgError, t.update, self.point1, singular_sigma, 5.3)
        self.assertRaises(np.linalg.LinAlgError, t.update, self.point2, asymmetric_sigma, 5.4)

        # check that the internal variables of ``g`` do not change after LinAlgError
        np.testing.assert_equal(t.sigma, offdiag_sigma)
        np.testing.assert_equal(t.mu, self.mean)
        self.assertEqual(t.dim, 2)
        self.assertEqual(t.dof, dof)
        np.testing.assert_equal(t.propose(1, fake_rng)[0], sample)
        np.testing.assert_equal(t.evaluate(sample), evaluate_at_sample)

    def test_dim_mismatch(self):
        mu    = np.ones(2)
        sigma = np.eye (3)
        dof   = 4.
        self.assertRaisesRegexp(AssertionError,
                'Dimensions of mean \(2\) and covariance matrix \(3\) do not match!',
                StudentT, mu, sigma, dof)

    def test_evaluate(self):
        self.assertAlmostEqual(self.t.evaluate(self.point1), self.target1, delta=self.delta)
        self.assertAlmostEqual(self.t.evaluate(self.point2), self.target2, delta=self.delta)

    def test_multi_evaluate(self):
        samples = np.array([self.point1 , self.point2 ])
        target  = np.array([self.target1, self.target2])

        out1 = np.empty(2)
        out2 = self.t.multi_evaluate(samples, out1)
        np.testing.assert_array_almost_equal(out1, target)
        np.testing.assert_array_almost_equal(out2, target)
        assert out1 is out2

        # should also work if out is not provided
        result = self.t.multi_evaluate(samples)
        np.testing.assert_array_almost_equal(result, target)

    def test_propose(self):
        mean  = np.array( [8.] )
        sigma = np.array([[.2]])
        dof   = 5.

        delta       = 0.005
        target_mean = mean
        target_cov  = dof / (dof - 2.) * sigma

        comp = StudentT(mu = mean, sigma = sigma, dof = dof)

        np.random.seed(rng_seed)

        # test if value for rng can be omitted
        proposed1 = comp.propose(rng_steps//2)
        # test if value for rng can be set
        proposed2 = comp.propose(rng_steps//2, np.random.mtrand)
        # test standard value for parameter N
        proposed3 = comp.propose()
        self.assertEqual(len(proposed3),1)

        proposed = np.vstack((proposed1, proposed2, proposed3))

        sampled_mean = proposed.mean(axis=0)
        sampled_cov  = np.cov(proposed, rowvar=0)

        self.assertAlmostEqual(sampled_mean, target_mean, delta=delta)
        self.assertAlmostEqual(sampled_cov , target_cov , delta=delta)
