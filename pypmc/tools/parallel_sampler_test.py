'''Unit tests for parallel sampler
In order to run tests in parallel, you have to execute this test with
"mpirun", for example: "mpirun -n 2 nosetests parallel_sampler_test.py"

'''

import numpy as np
from nose.plugins.attrib import attr
from ..sampler.markov_chain import MarkovChain, AdaptiveMarkovChain
from ..sampler.importance_sampling import ImportanceSampler
from ..density.mixture_test import DummyComponent
from .. import density
from ._probability_densities import unnormalized_log_pdf_gauss

def setUpModule():
    try:
        from mpi4py import MPI
    except ImportError:
        raise unittest.SkipTest("Cannot test MPI parallelism without MPI4Py")

    from .parallel_sampler import MPISampler, MPI

    global MPI
    global MPISampler
    global comm
    global rank
    global size
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

import unittest

rng_seed = 215195153

target_mean      = np.array([4.3, 1.1])
target_sigma     = np.array([[0.01 , 0.003 ]
                            ,[0.003, 0.0025]])
inv_target_sigma = np.linalg.inv(target_sigma)
log_target = lambda x: unnormalized_log_pdf_gauss(x, target_mean, inv_target_sigma)

class TestMPISampler(unittest.TestCase):
    def setUp(self):
        np.random.mtrand.seed(rng_seed)

    @attr('slow')
    def test_mc_self_adaptive_sampling(self):
        NumberOfRandomSteps = 20000
        delta_mean = .01
        delta_cov  = .05

        prop_dof   = 5.
        prop_sigma = np.array([[0.01, 0.   ]
                              ,[0.  , 0.001]])

        prop = density.student_t.LocalStudentT(prop_sigma, prop_dof)

        #extremely bad starting values
        start = np.array([-3.7, 10.6])

        # should be able to create an MPISampler with these syntaxes
        psampler = MPISampler(AdaptiveMarkovChain, MPI.COMM_WORLD, log_target, prop, start, prealloc=NumberOfRandomSteps)
        psampler = MPISampler(AdaptiveMarkovChain, target=log_target, proposal=prop, start=start)

        self.assertEqual(len(psampler.sampler.samples), 0)
        if rank == 0:
            self.assertEqual(len(psampler.samples_list), size)
            for history_instance in psampler.samples_list:
                self.assertEqual(len(history_instance), 0)
            self.assertTrue(psampler.weights_list is None)
        else:
            self.assertTrue(psampler.samples_list is None)
            self.assertTrue(psampler.weights_list is None)

        # prerun for burn-in
        psampler.run(NumberOfRandomSteps//10)

        self.assertEqual(len(psampler.sampler.samples), 1)
        if rank == 0:
            self.assertEqual(len(psampler.samples_list), size)
            for history_instance in psampler.samples_list:
                self.assertEqual(len(history_instance), 1)
                self.assertEqual(len(history_instance[-1]), NumberOfRandomSteps//10)
            self.assertTrue(psampler.weights_list is None)
        else:
            self.assertTrue(psampler.samples_list is None)
            self.assertTrue(psampler.weights_list is None)

        psampler.clear()

        self.assertEqual(len(psampler.sampler.samples), 0)
        if rank == 0:
            self.assertEqual(len(psampler.samples_list), size)
            for history_instance in psampler.samples_list:
                self.assertEqual(len(history_instance), 0)
        else:
            self.assertTrue(psampler.samples_list is None)

        for i in range(10):
            psampler.run(NumberOfRandomSteps//10)
            # Note: each process only uses its own samples for proposal adaptation
            psampler.sampler.adapt()

        process_own_values = psampler.sampler.samples[:]

        sample_mean = process_own_values.mean(axis=0)
        sample_cov  = np.cov(process_own_values, rowvar=0)

        np.testing.assert_allclose(sample_mean, target_mean , delta_mean)
        np.testing.assert_allclose(sample_cov , target_sigma, delta_cov )

        if rank == 0:
            gathered_samples = [history_item[:] for history_item in psampler.samples_list]
            # each process should have produced exactly the same samples because of
            # exactly the same random seed
            for samples in gathered_samples:
                np.testing.assert_equal(samples, gathered_samples[0])

            gathered_samples = np.vstack(gathered_samples)
            self.assertEqual(gathered_samples.shape, (NumberOfRandomSteps * size, 2) )

            all_samples_mean = gathered_samples.mean(axis=0)
            all_samples_cov  = np.cov(gathered_samples, rowvar=0)

            np.testing.assert_allclose(sample_mean, target_mean , delta_mean)
            np.testing.assert_allclose(sample_cov , target_sigma, delta_cov )

    def test_run_return_value(self):
        NumberOfRandomSteps = 100

        dummy_prop   = density.mixture.MixtureDensity( [DummyComponent(propose=[float(i)]) for i in range(5)] )
        dummy_target = lambda x: 0.

        psampler = MPISampler(ImportanceSampler, target=dummy_target, proposal=dummy_prop)

        run_output = psampler.run(NumberOfRandomSteps, trace_sort=True)

        if rank != 0:
            for i in range(5):
                assert float(i) in run_output
                self.assertEqual(float(run_output[i]), psampler.sampler.samples[:][i][0])
        else:
            self.assertEqual(len(run_output), size)
            for process_id in range(size):
                self.assertEqual( len(run_output[process_id]), NumberOfRandomSteps )
                for sample_index in range(NumberOfRandomSteps):
                    self.assertEqual(psampler.samples_list[process_id][:][sample_index][0],
                                     float(run_output[process_id][sample_index])           )
