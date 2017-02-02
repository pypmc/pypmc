"""Unit tests for the mixture probability densities

"""

from .mixture import *
import numpy as np
import unittest

rng_seed  = 12850419774 % 4294967296
rng_steps = 50000

# dummy proposal component (convenient for testing):
#   - evaluates to exactly the input
#   - proposes always the same
class DummyComponent(ProbabilityDensity):
    def __init__(self, propose = [0.], eval_to = 42.):
        self.to_propose = np.array(propose)
        self.dim = len(self.to_propose)
        self.eval_to = eval_to
    def evaluate(self, x):
        return self.eval_to
    def propose(self, N=1):
        return np.array([self.to_propose for i in range(N)])

class TestMixtureDensity(unittest.TestCase):
    ncomp          = 5
    components     = [DummyComponent   for i in range(ncomp)]

    proposals   = (DummyComponent(eval_to=10.),DummyComponent())
    weights     = (.9,.1)
    evaluate_at = np.array( (-5.,) )

    target = 39.69741490700607

    mix = MixtureDensity(proposals, weights)

    def setUp(self):
        np.random.seed(rng_seed)

    def test_dimcheck(self):
        # dimensions of all components have to match

        # create instances
        comp_instances = [DummyComponent() for i in range(self.ncomp)]

        # should not raise an error
        MixtureDensity(comp_instances)

        # blow one dim
        comp_instances[2].dim = 100

        with self.assertRaises(AssertionError):
            MixtureDensity(comp_instances)

    def test_normalize(self):
        mix = MixtureDensity(self.components)

        # automatic normalization
        self.assertTrue(mix.normalized())
        np.testing.assert_allclose(mix.weights, 1. / self.ncomp, rtol = 1.e-15) # double precision

        # blow normalization
        mix.weights[0] = 2
        self.assertFalse(mix.normalized())

        # renormalize
        mix.normalize()
        self.assertTrue(mix.normalized())

    def test_prune(self):
        # range(self.ncomp) makes the first weight equal to zero
        mix = MixtureDensity(self.components, range(self.ncomp))

        # removing elements
        self.assertEqual(mix.prune(), [(0,DummyComponent,0.)])
        self.assertEqual(len(mix.weights), self.ncomp - 1)
        self.assertTrue(mix.normalized())

    def test_evaluate(self):
        self.assertAlmostEqual(self.target, self.mix.evaluate(self.evaluate_at))

    def test_multi_evaluate(self):
        samples = np.array([self.evaluate_at] * 2)
        targets = np.array([self.target] * 2)
        individual = np.zeros((2,2))
        out1 = np.zeros(2)
        out2 = np.zeros(2)
        res1 = self.mix.multi_evaluate(samples, individual=individual)
        res2 = self.mix.multi_evaluate(samples, individual=individual, out=out1)
        res3 = self.mix.multi_evaluate(samples, out=out2)

        # assert bitwise equality no matter where the result is taken from or what is calculated in addition
        np.testing.assert_equal(res1, res2)
        np.testing.assert_equal(res1, res3)
        np.testing.assert_equal(res1, out1)
        np.testing.assert_equal(res1, out2)

        np.testing.assert_array_almost_equal(res1, targets)
        np.testing.assert_array_almost_equal(res2, targets)
        np.testing.assert_array_almost_equal(res3, targets)
        np.testing.assert_array_almost_equal(out1, targets)
        np.testing.assert_array_almost_equal(out2, targets)
        np.testing.assert_array_almost_equal(individual[:,0], 10.)
        np.testing.assert_array_almost_equal(individual[:,1], 42.)

    def test_error_messages_multi_evaluate(self):
        samples              = np.array([[1.], [2.], [3.]])
        samples_wrong_dim    = np.array([[1., 1.2], [2. ,32.], [2, 3.]])
        individual_ok        = np.empty((3,2))
        individual_too_short = np.empty((2,2))
        individual_wrong_K   = np.empty((3,3))
        out_too_long         = np.empty((9, ))
        out_ok               = np.empty((3, ))
        components           = [0]

        self.mix.multi_evaluate(samples, individual=individual_ok) # should be ok
        self.assertRaisesRegexp(AssertionError, 'x.*wrong dim.*',
                                self.mix.multi_evaluate, samples_wrong_dim, individual=individual_ok)
        self.assertRaisesRegexp(AssertionError, 'individual.*must.*shape',
                                self.mix.multi_evaluate, samples, individual=individual_too_short)
        self.assertRaisesRegexp(AssertionError, 'individual.*must.*shape',
                                self.mix.multi_evaluate, samples, individual=individual_wrong_K)
        self.assertRaisesRegexp(AssertionError, 'components.*not None.*out.*must be None',
                                self.mix.multi_evaluate, samples, out_ok, components=components)
        self.assertRaisesRegexp(AssertionError, 'out.*must.*len.*3',
                                self.mix.multi_evaluate, samples, out_too_long)

    def test_propose(self):
        np.random.seed(rng_seed)

        proposes    = np.array( ((-5.,),(+5.,)) )
        weights     = (.8,.2)
        proposals   = (DummyComponent(proposes[0]),DummyComponent(proposes[1]))
        delta       = 2. * np.sqrt(np.cov(proposes.reshape(1,2),ddof=0) / rng_steps) # 2-sigma-interval

        mix = MixtureDensity(proposals, weights)
        # mix should propose values from ``proposals[0]`` with abundance 80% and
        # from ``proposals[1]`` with abundance 20% (i.e. according to ``weights``)

        occurrences = np.empty(2)
        values1 = mix.propose(rng_steps//2) # test if value for rng can be omitted
        values2 = mix.propose(rng_steps//2, np.random) # test if value for rng can be set
        occurrences[0] = (values1 <  0.).sum() + (values2 <  0.).sum()
        occurrences[1] = (values1 >= 0.).sum() + (values2 >= 0.).sum()
        abundances = occurrences/float(rng_steps)

        self.assertAlmostEqual(abundances[0], weights[0], delta = delta)
        self.assertAlmostEqual(abundances[1], weights[1], delta = delta)

    def test_tracing_propose(self):
        # test if the mixture proposal correctly traces the responsible component
        components = []
        for i in range(5):
            components.append( DummyComponent(propose=[float(i)]) )
        prop = MixtureDensity(components)
        samples, origins = prop.propose(50, trace=True, shuffle=False)
        for i in range(50):
            self.assertAlmostEqual(samples[i], origins[i], delta=1.e-15)

    def test_shuffle(self):
        components = [DummyComponent(propose=[-1., 0.]), DummyComponent(propose=[+1.,5.])]
        prop = MixtureDensity(components)
        samples = prop.propose(50, shuffle=True)

        print(samples)
        # make sure there is "+1" and "-1" within the first few samples
        self.assertAlmostEqual(samples[0][0], -1., delta=1.e-15)
        self.assertAlmostEqual(samples[1][0], +1., delta=1.e-15)

    def test_no_segfault(self):
        # passing an empty list as ``components`` used to cause segfault
        self.assertRaisesRegexp(AssertionError, ".*at least.*['one''1'].*component", MixtureDensity, [])

means = np.array([[ 1.0,  5.4, -3.1],
                  [-3.8,  2.5,  0.4],
                  [ 4.1, -3.3, 19.8],
                  [-9.1, 25.4,  1.0]])

covs  = np.array([[[ 3.7,  0.7, -0.6],
                   [ 0.7,  4.5,  0.5],
                   [-0.6,  0.5,  0.6]],

                  [[ 7.0,  1.2,  0.6],
                   [ 1.2,  1.3,  1.5],
                   [ 0.6,  1.5,  4.1]],

                  [[ 1.3,  0.9, -0.3],
                   [ 0.9,  4.1, -0.2],
                   [-0.3, -0.2,  2.2]],

                  [[ 1.6, -0.3, -0.6],
                   [-0.3,  6.6, -0.5],
                   [-0.6, -0.5,  9.4]]])

dofs = range(1,5)

weights = np.array([ 2.7,  0.4, 1.6, 4.8])

normalized_weights = weights/weights.sum()

class TestCreateGaussian(unittest.TestCase):
    def test_invalid_input(self):
        self.assertRaisesRegexp(AssertionError, 'Number of means.*?not match.*?number of cov',
                                create_gaussian_mixture, means    , covs[:2]   )
        self.assertRaisesRegexp(AssertionError, 'Number of means.*?not match.*?number of cov',
                                create_gaussian_mixture, means[:2], covs       )

    def test_create_no_weights(self):
        mix = create_gaussian_mixture(means, covs)

        self.assertEqual(len(mix.components), 4)
        self.assertEqual(len(mix.weights)   , 4)

        for i in range(4):
            self.assertAlmostEqual(mix.weights[i], .25)
            np.testing.assert_equal(mix.components[i].mu   , means[i])
            np.testing.assert_equal(mix.components[i].sigma, covs [i])

    def test_create_with_weights(self):
        mix = create_gaussian_mixture(means, covs, weights)

        self.assertEqual(len(mix.components), 4)
        self.assertEqual(len(mix.weights)   , 4)

        for i in range(4):
            self.assertAlmostEqual(mix.weights[i], normalized_weights[i])
            np.testing.assert_equal(mix.components[i].mu   , means[i])
            np.testing.assert_equal(mix.components[i].sigma, covs [i])

class TestRecoverGaussian(unittest.TestCase):
    def setUp(self):
        print('when this test fails, first make sure that "create_gaussian_mixture" works')

    def test_recover(self):
        mix = create_gaussian_mixture(means, covs, weights)
        o_means, o_covs, o_weights = recover_gaussian_mixture(mix)
        for i in range(4):
            self.assertAlmostEqual (o_weights[i], mix.weights   [i]      )
            np.testing.assert_equal(o_means  [i], mix.components[i].mu   )
            np.testing.assert_equal(o_covs   [i], mix.components[i].sigma)

class TestCreateStudentT(unittest.TestCase):
    def test_invalid_input(self):
        self.assertRaisesRegexp(AssertionError, 'Number of.*?means.*?covs.*?dofs.*?not match.',
                                create_t_mixture, means    , covs[:2], dofs       )
        self.assertRaisesRegexp(AssertionError, 'Number of.*?means.*?covs.*?dofs.*?not match.',
                                create_t_mixture, means[:2], covs    , dofs       )
        self.assertRaisesRegexp(AssertionError, 'Number of.*?means.*?covs.*?dofs.*?not match.',
                                create_t_mixture, means[:2], covs    , dofs[:2]   )

    def test_create_no_weights(self):
        mix = create_t_mixture(means, covs, dofs)

        self.assertEqual(len(mix.components), 4)
        self.assertEqual(len(mix.weights)   , 4)

        for i in range(4):
            self.assertAlmostEqual(mix.weights[i], .25)
            np.testing.assert_equal(mix.components[i].mu   , means[i])
            np.testing.assert_equal(mix.components[i].sigma, covs [i])
            np.testing.assert_equal(mix.components[i].dof  , dofs [i])

    def test_create_with_weights(self):
        mix = create_t_mixture(means, covs, dofs, weights)

        self.assertEqual(len(mix.components), 4)
        self.assertEqual(len(mix.weights)   , 4)

        for i in range(4):
            self.assertAlmostEqual(mix.weights[i], normalized_weights[i])
            np.testing.assert_equal(mix.components[i].mu   , means[i])
            np.testing.assert_equal(mix.components[i].sigma, covs [i])
            np.testing.assert_equal(mix.components[i].dof  , dofs [i])

class TestRecoverStudentT(unittest.TestCase):
    def setUp(self):
        print('when this test fails, first make sure that "create_gaussian_mixture" works')

    def test_recover(self):
        mix = create_t_mixture(means, covs, dofs, weights)
        o_means, o_covs, o_dofs, o_weights = recover_t_mixture(mix)
        for i in range(4):
            self.assertAlmostEqual (o_weights[i], mix.weights   [i]      )
            np.testing.assert_equal(o_means  [i], mix.components[i].mu   )
            np.testing.assert_equal(o_covs   [i], mix.components[i].sigma)
            np.testing.assert_equal(o_dofs   [i], mix.components[i].dof  )
