"""Unit tests for Variational Bayes.

"""

from .variational import *
import numpy as np
import unittest

invertible_matrix = np.array([[2.   , 3.    ],
                              [2.   , 2.   ]])

covariance1       = np.array([[0.01 , 0.003 ],
                              [0.003, 0.0025]])

covariance2       = np.array([[0.1  , 0.    ],
                              [0.   , 0.02  ]])

float64_acc = 1.e-15

delta_abun = .01
delta_mean = .5
delta_cov0 = .0002
delta_cov1 = .001


rng_seed = 625135153

class TestGaussianInference(unittest.TestCase):
    def test_update(self):
        np.random.mtrand.seed(rng_seed)

        # generate test data from two independent gaussians
        target_abundances = .5
        target_mean1 = np.array((+5. , 0.))
        target_mean2 = np.array((-5. , 0.))
        target_cov1  = covariance1
        target_cov2  = covariance2
        data1 = np.random.mtrand.multivariate_normal(target_mean1, target_cov1, size = 10**4)
        data2 = np.random.mtrand.multivariate_normal(target_mean2, target_cov2, size = 10**4)
        test_data = np.vstack((data1,data2))
        np.random.shuffle(test_data)

        # provide hint for means to force convergence to a specific solution
        infer = GaussianInference(test_data, 2, m0 = np.vstack((target_mean1-2.,target_mean2+2.)) )
        infer.update()
        inferred_abundances, inferred_means, inferred_covars = infer.get_result()

        self.assertAlmostEqual(target_abundances, inferred_abundances[0], delta_abun)
        self.assertAlmostEqual(target_abundances, inferred_abundances[1], delta_abun)

        self.assertTrue( (np.abs(target_mean1 - inferred_means[0])<delta_mean).all()  )
        self.assertTrue( (np.abs(target_mean2 - inferred_means[1])<delta_mean).all()  )

        self.assertTrue( (np.abs(target_cov1 - inferred_covars[0])<delta_cov0).all()  )
        self.assertTrue( (np.abs(target_cov2 - inferred_covars[1])<delta_cov1).all()  )

    def test_set_variational_parameters(self):
        infer = GaussianInference(np.empty((20,20)), 5, nu0 = 0.)

        self.assertEqual(infer.nu0, 0.)

        infer.set_variational_parameters(beta0 = 2., W0 = invertible_matrix)

        # set_variational_parameters shall reset not passed parameters to default
        self.assertNotEqual(infer.nu0   , 0.)
        self.assertEqual   (infer.beta0 , 2.)
        self.assertTrue    ( (np.abs(infer.inv_W0-np.linalg.inv(invertible_matrix))<float64_acc).all() )

    def test_prune(self):
        np.random.mtrand.seed(rng_seed)

        # generate test data from two independent gaussians
        target_abundances = np.array((.1, .9))
        target_mean1 = np.array((+1. , -4.))
        target_mean2 = np.array((-5. , +2.))
        target_cov1  = covariance1
        target_cov2  = covariance2
        data1 = np.random.mtrand.multivariate_normal(target_mean1, target_cov1, size =   10**3)
        data2 = np.random.mtrand.multivariate_normal(target_mean2, target_cov2, size = 9*10**3)
        test_data = np.vstack((data1,data2))
        np.random.shuffle(test_data)

        # provide hint for means to force convergence to a specific solution
        infer = GaussianInference(test_data, 3, m0 = np.vstack((target_mean1+2.,target_mean2+2.,
                                                        np.zeros_like(target_mean1)              )) )
        infer.update()
        infer.prune()
        infer.update()
        inferred_abundances, inferred_means, inferred_covars = infer.get_result()

        # the additional component should have been pruned out
        self.assertEqual(len(inferred_abundances),2)

        self.assertAlmostEqual(target_abundances[0], inferred_abundances[0], delta_abun)
        self.assertAlmostEqual(target_abundances[1], inferred_abundances[1], delta_abun)

        self.assertTrue( (np.abs(target_mean1 - inferred_means[0])<delta_mean).all()  )
        self.assertTrue( (np.abs(target_mean2 - inferred_means[1])<delta_mean).all()  )

        self.assertTrue( (np.abs(target_cov1 - inferred_covars[0])<delta_cov0).all()  )
        self.assertTrue( (np.abs(target_cov2 - inferred_covars[1])<delta_cov1).all()  )
