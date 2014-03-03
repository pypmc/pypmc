"""Unit tests for the PMC proposal functions.

"""

from .proposal import *
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

class TestMixtureProposal(unittest.TestCase):
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

class TestGaussianComponent(unittest.TestCase):
    def setUp(self):
        print('"GaussianComponent" is a wrapper of ..markov_chain.proposal.MultivariateGaussian.')
        print('When this test fails, first make sure that ..markov_chain.proposal.MultivariateGaussian works.')

    def test_dim_mismatch(self):
        mu    = np.ones(2)
        sigma = np.eye (3)
        self.assertRaisesRegexp(AssertionError, 'Dimensions of mean \(2\) and covariance matrix \(3\) do not match!', Gauss, mu, sigma)

    def test_evaluate(self):
        sigma = np.array([[0.01 , 0.003 ]
                         ,[0.003, 0.0025]])

        delta = 1e-8

        mean  = np.array([4.3 , 1.1])
        point = np.array([4.35, 1.2])

        comp = Gauss(mean, sigma=sigma)

        target = 1.30077135

        self.assertAlmostEqual(comp.evaluate(point), target, delta=delta)

    def test_propose(self):
        mean           = np.array([-3.   ,+3.    ])

        offdiag_sigma  = np.array([[0.01 , 0.003 ]
                                  ,[0.003, 0.0025]])

        delta_mean   = .001
        delta_cov_00 = .0001
        delta_cov_01 = .00003
        delta_cov_11 = .00003

        comp = Gauss(mu=mean, sigma=offdiag_sigma)

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
        sampled_cov  = np.cov(proposed,rowvar=0)


        self.assertAlmostEqual(sampled_mean[0], mean[0], delta=delta_mean)
        self.assertAlmostEqual(sampled_mean[1], mean[1], delta=delta_mean)

        self.assertAlmostEqual(sampled_cov[0,0] , offdiag_sigma[0,0] , delta=delta_cov_00)
        self.assertAlmostEqual(sampled_cov[0,1] , offdiag_sigma[0,1] , delta=delta_cov_01)
        self.assertAlmostEqual(sampled_cov[1,1] , offdiag_sigma[1,1] , delta=delta_cov_11)

class TestStudentTComponent(unittest.TestCase):
    def setUp(self):
        print('"StudentTComponent" is a wrapper of ..markov_chain.proposal.MultivariateStudentT.')
        print('When this test fails, first make sure that ..markov_chain.proposal.MultivariateStudentT works.')

    def test_dim_mismatch(self):
        mu    = np.ones(2)
        sigma = np.eye (3)
        dof   = 4.
        self.assertRaisesRegexp(AssertionError,
                'Dimensions of mean \(2\) and covariance matrix \(3\) do not match!',
                StudentT, mu, sigma, dof)

    def test_evaluate(self):
        mean  = np.array( [1.25, 4.3   ] )
        sigma = np.array([[0.0049, 0.  ]
                         ,[0.    ,  .01]])
        dof   = 5.
        delta = 1e-9

        t = StudentT(mean, sigma, dof)

        point1 = np.array([1.3 , 4.4  ])
        point2 = np.array([1.26, 4.424])

        target1 = 2.200202941
        target2 = 2.174596526

        self.assertAlmostEqual(t.evaluate(point1), target1, delta=delta)
        self.assertAlmostEqual(t.evaluate(point2), target2, delta=delta)

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
