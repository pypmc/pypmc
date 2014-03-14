"""Unit tests for the mixture probability densities

"""

from .mixture import *
import numpy as np
import unittest

rng_seed  = 12850419774
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

    def test_access(self):
        mix = MixtureDensity(self.components, np.arange(self.ncomp))

        normalized_weights  = np.arange(self.ncomp, dtype=float)
        normalized_weights /= normalized_weights.sum()

        target_output = [(self.components[i],normalized_weights[i]) for i in range(self.ncomp)]

        # test iteration
        iter_output = []
        for tup in mix:
            iter_output.append(tup)
        self.assertEqual(iter_output,target_output)

        # test item access
        item_output = [mix[i] for i in range(self.ncomp)]
        self.assertEqual(item_output,target_output)

    def test_evaluate(self):
        proposals   = (DummyComponent(eval_to=10.),DummyComponent())
        weights     = (.9,.1)
        evaluate_at = np.array( (-5.,) )

        target = 39.69741490700607

        mix = MixtureDensity(proposals, weights)

        self.assertAlmostEqual(target,mix.evaluate(evaluate_at))

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
        values1 = mix.propose(rng_steps/2) # test if value for rng can be omitted
        values2 = mix.propose(rng_steps/2, np.random) # test if value for rng can be set
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

        # make sure there is "+1" and "-1" within the first few samples
        self.assertAlmostEqual(samples[0][0], -1., delta=1.e-15)
        self.assertAlmostEqual(samples[1][0], +1., delta=1.e-15)

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

weights = np.array([ 2.7,  0.4, 1.6, 4.8])

normalized_weights = weights/weights.sum()

class TestCreateGaussian(unittest.TestCase):
    def test_invalid_input(self):
        self.assertRaisesRegexp(AssertionError, 'number of means.*?not match.*?number of cov',
                                create_gaussian_mixture, means    , covs[:2]   )
        self.assertRaisesRegexp(AssertionError, 'number of means.*?not match.*?number of cov',
                                create_gaussian_mixture, means[:2], covs       )

    def test_create_no_weights(self):
        mix = create_gaussian_mixture(means, covs)

        for i, (component, weight) in enumerate(mix):
            self.assertAlmostEqual(weight, .25)
            np.testing.assert_equal(component.mu   , means[i])
            np.testing.assert_equal(component.sigma, covs [i])

    def test_create_with_weights(self):
        mix = create_gaussian_mixture(means, covs, weights)

        for i, (component, weight) in enumerate(mix):
            self.assertAlmostEqual(weight, normalized_weights[i])
            np.testing.assert_equal(component.mu   , means[i])
            np.testing.assert_equal(component.sigma, covs [i])

class TestRecoverGaussian(unittest.TestCase):
    def setUp(self):
        print('when this test fails, first make sure that "create_gaussian_mixture" works')

    def test_recover(self):
        mix = create_gaussian_mixture(means, covs, weights)
        o_means, o_covs, o_weights = recover_gaussian_mixture(mix)
        for i, (component, weight) in enumerate(mix):
            self.assertAlmostEqual(o_weights[i], weight)
            np.testing.assert_equal(o_means[i], component.mu   )
            np.testing.assert_equal(o_covs [i], component.sigma)