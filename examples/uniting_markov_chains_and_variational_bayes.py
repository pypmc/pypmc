'''This example illustrates how pypmc can be used to integrate a
non-negative function. The presented algorithm needs very little
analytical knowledge about the function.

'''

import numpy as np
import pypmc

# The idea is to find a good proposal function for importance sampling
# with as little information about the target function as possible.
#
# In this example we will first map out regions of interest using Markov
# chains, then we use the variational Bayes to approximate the target
# with a Gaussian mixture.

# *************************** Important: ***************************
# * The target function must be defined such that it returns the   *
# * log of the function of interest. The methods we use imply that *
# * the function is interpreted as an unnormalized probability     *
# * density.                                                       *
# ******************************************************************

# Define the target; i.e., the function you want to sample from.  In
# this case, it is a Student's t mixture of three components with
# different degrees of freedom. They are located close to each other.
# If you want a multimodal target, adjust the means.

dim = 2

mean0       = np.array ([-6.0,  7.3  ])
covariance0 = np.array([[ 0.8, -0.3 ],
                        [-0.3,  1.25]])

mean1       = np.array ([-7.0,  8.0   ])
covariance1 = np.array([[ 0.5,  0.    ],
                        [ 0. ,  0.2  ]])

mean2       = np.array ([-8.5,  7.5   ])
covariance2 = np.array([[ 0.5,  0.2   ],
                        [ 0.2,  0.2  ]])

component_weights = np.array([0.3, 0.4, 0.3])
component_means = [mean0, mean1, mean2]
component_covariances = [covariance0, covariance1, covariance2]
dofs = [13, 17, 5]

target_mixture = pypmc.density.mixture.create_t_mixture(component_means, component_covariances, dofs, component_weights)
log_target = target_mixture.evaluate

# Now we suppose that we only have the following knowledge about the
# target function: its regions of interest are at a distance of no more
# than order ten from zero.

# Now we try to find these with Markov chains.  We have to deal with
# the fact that there may be modes separated by regions of very low
# probability. It is thus unlikely that a single chain explores more
# than one mode in such a case. To deal with this multimodality, we
# start several chains and hope that they find all modes.  We will
# start ten Markov chains at random positions in the square
# [(-10,-10), (+10,+10)].
starts = [np.random.uniform(-10,10, size=dim) for i in range(10)]

# For a local-random-walk Markov chain, we also need an initial
# proposal.  Here, we take a gaussian with initial covariance
# diag(1e-3).  The initial covariance should be chosen such that it is
# of the same order as the real covariance of the mode to be mapped
# out. For a Gaussian target, the overall scale should
# decrease as 2.38^2/d as the dimension d increases to achieve an
# acceptance rate around 20%.
mc_prop = pypmc.density.gauss.LocalGauss(np.eye(dim) * 2.38**2 / dim)
mcs = [pypmc.sampler.markov_chain.AdaptiveMarkovChain(log_target, mc_prop, start) for start in starts]

print('running Markov chains ...')

# In general we need to let the chain move to regions of high
# probability, these samples are not representative, so we discard them
# as burn-in. Then we let the Markov chains map out the regions of
# interest. The samples are used to adapt the proposal covariance to
# yield a satisfactory acceptance rate.
for mc in mcs:
    for i in range(20):
        mc.run(500)
        mc.adapt()
        if i == 0:
            mc.clear()

mc_samples_sorted_by_chain = [mc.samples[:] for mc in mcs]
mc_samples = np.vstack(mc_samples_sorted_by_chain)

means = np.zeros((len(mcs), dim))
variances = np.zeros_like(means)

for i,mc in enumerate(mc_samples_sorted_by_chain):
    means[i] = mc.mean(axis=0)
    variances[i] = mc.var(axis=0)

# Now we use the Markov chain samples to generate a mixture proposal
# function for importance sampling. For this purpose, we choose the
# variational Bayes algorithm that takes samples and an initial guess
# of the mixture as input. To create the initial guess, we group all
# chains that mixed, and create 10 components per group. For a
# unimodal target, all chains should mix. For more information about
# the following call, check the example "Grouping by Gelman-Rubin R
# value"(r-group.py) or the reference documentation.
long_patches = pypmc.mix_adapt.r_value.make_r_gaussmix(mc_samples_sorted_by_chain, K_g=10)

# Comments on arguments:
#   o mc_samples[::100]   - Samples in the Markov chains are strongly correlated
#                           => thin the samples to get approx. independent samples
#   o W0=np.eye(dim)*1e10 - The resulting covariance matrices can be very
#                           sensitive to W0. Its inverse should be chosen much
#                           smaller than the actual covariance. If it is too small,
#                           W0 will dominate the resulting covariances and
#                           usually lead to very bad results.
vb = pypmc.mix_adapt.variational.GaussianInference(mc_samples[::100], initial_guess=long_patches, W0=np.eye(dim)*1e10)

# When we run variational Bayes, we want unneccessary components to be
# automatically pruned. The prune parameter sets how many samples a
# component must effectively have to be considered important. The rule
# of thumb employed here proved good in our experiments.
vb_prune = 0.5 * len(vb.data) / vb.K

# Run the variational Bayes for at most 1,000 iterations.  But if the
# lower bound of the model evidence changes by less than `rel_tol`,
# convergence is declared before. If we increase `rel_tol` to 1e-4, it
# takes less iterations but potentially more (useless) components
# survive the pruning. The trade-off depends on the complexity of the
# problem.
print('running variational Bayes ...')
vb.run(1000, rel_tol=1e-8, abs_tol=1e-5, prune=vb_prune, verbose=True)

# extract the most probable Gaussian mixture given the samples
vbmix = vb.make_mixture()

# Now we can instantiate an importance sampler. We draw 1,000
# importance samples and use these for a proposal update using
# variational Bayes again. In case there are multiple modes and
# the chains did not mix, we need this step to infer the right
# component weights because the component weight is given by
# how many chains it attracted, which could be highly dependent
# on the starting points and independent of the correct
# probability mass.
print('running importance sampling ...')
sampler = pypmc.sampler.importance_sampling.ImportanceSampler(log_target, vbmix)
sampler.run(1000)

# The variational Bayes allows us, unlike PMC, to include the
# information gained by the Markov chains in subsequent proposal
# updates. We know that we cannot trust the component weights obtained
# by the chains. Nevertheless, we can rely on the means and covariances. The
# following lines show how to code that into the variational Bayes by
# removing the hyperparameter `alpha0` that encodes the component
# weights.
prior_for_proposal_update = vb.posterior2prior()
prior_for_proposal_update.pop('alpha0')
vb2 = pypmc.mix_adapt.variational.GaussianInference(sampler.samples[:],
                                                    initial_guess=vbmix,
                                                    weights=sampler.weights[:][:,0],
                                                    **prior_for_proposal_update)

# Note: This time we leave "prune" at the default value "1" because we
#       want to keep all components that are expected to contribute
#       with at least one effective sample per importance sampling run.
print('running variational Bayes ...')
vb2.run(1000, rel_tol=1e-8, abs_tol=1e-5, verbose=True)
vb2mix = vb2.make_mixture()

# Now we draw another 10,000 samples with the updated proposal
sampler.proposal = vb2mix
print('running importance sampling ...')
sampler.run(10**4)

# We can combine the samples and weights from the two runs, see reference [Cor+12].
weights = pypmc.sampler.importance_sampling.combine_weights([samples[:]      for samples in sampler.samples],
                                                            [weights[:][:,0] for weights in sampler.weights],
                                                            [vbmix, vb2mix]                                 ) \
                                                            [:][:,0]
samples = sampler.samples[:]

# The integral can then be estimated from the weights. The error is also
# estimated from the weights. By the central limit theorem, the integral
# estimator has a gaussian distribution.
integral_estimator = weights.sum() / len(weights)
integral_uncertainty_estimator = np.sqrt((weights**2).sum() / len(weights) - integral_estimator**2) / np.sqrt(len(weights)-1)

print('analytical integral = 1')
print('estimated  integral =', integral_estimator, '+-', integral_uncertainty_estimator)

# Let's see how good the proposal matches the target density: the closer
# the values of perplexity and effective sample size (ESS) are to 1,
# the better.  Outliers, or samples out in the tails of the target
# with a very large weight, show up in the 2D marginal and reduce the
# ESS significantly. For the above integral estimate to be right on
# average, they are 'needed'. Without outliers (most of the time), the
# integral is a tad too small.
print('perplexity', pypmc.tools.convergence.perp(weights))
print('effective sample size', pypmc.tools.convergence.ess(weights))

# As mentioned in the very beginning, the methods we applied reinterpret
# the target function as an unnormalized probability density.
# In addition to the integral, we also get weighted samples distributed according
# to that probability density.
try:
    import matplotlib.pyplot as plt
except ImportError:
    print('For plotting "matplotlib" needs to be installed')
    exit(1)

plt.figure()
plt.hist2d(samples[:,0], samples[:,1], weights=weights, bins=100, cmap='gray_r')
pypmc.tools.plot_mixture(sampler.proposal, visualize_weights=True, cmap='jet')
plt.colorbar()
plt.clim(0.0, 1.0)
plt.title('colors visualize component weights')
plt.show()
