'''This example illustrates how pypmc can be used to integrate a
non-negative function. The presented algorithm needs very little
analytical knowledge about the function.

'''

from __future__ import print_function
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

# define the target; i.e., the function you want to importance sample.
# In this case, it is the indicator function of four circles with radius
# 0.1 and centers [(-5,-5), (-5,+5), (+5,-5), (+5,+5)].
# The integral is then just the area of four circles with radius 0.1,
# i.e. four pi time 0.1**2.
target_integral = 4 * np.pi * 0.1**2
dim = 2

# We will use pypmc's indicator tools to implement these circles.
indicators = [
                  pypmc.tools.indicator.ball(center=(-1.0,-1.0), radius=0.1),
                  pypmc.tools.indicator.ball(center=(-1.0,+1.0), radius=0.1),
                  pypmc.tools.indicator.ball(center=(+1.0,-1.0), radius=0.1),
                  pypmc.tools.indicator.ball(center=(+1.0,+1.0), radius=0.1)
             ]
def log_target(x):
    # evaluate the individual indicator functions
    # if any returns True, the target is one, i.e. the log is zero
    for indicator in indicators:
        if indicator(x):
            return 0.0
    # this command is only reached when no indicator returned True
    # In that case the targt function is zero, i.e. the log is minus infinity
    return -np.inf

# Now, we suppose that we only have the following knowledge about the
# target function: Its regions of interest is at a distance of no more
# than order ten from zero.

# Now we try to find these with Markov chains.
# We always have to assume that there may be modes separated by regions
# of zero probability mass. It is thus unlikely that a single chain
# explores more than one mode in such a case. To deal with this
# multimodality, we start several chains and hope that they find all
# modes.
# We will start twenty Markov chains at random positions in the square
# [(-10,-10), (+10,+10)].
starts = [np.random.rand(dim) * 20 - 10 for i in range(20)]

# A Markov chain needs an initial point where the target is not exactly zero
for i in range(len(starts)):
    # draw a new point as long as we are outside the circles
    while log_target(starts[i]) == -np.inf:
        starts[i] = np.random.rand(dim) * 20 - 10

# For a local random walk Markov chain, we also need an initial proposal.
# Here, we take a gaussian with initial covariance diag(1e-3).
# The initial covariance should be chosen such that it is of the same order
# as the real covariance of the mode to be mapped out or less. The closer
# the covariance placed here is to the real covariance, the better the result.
mc_prop = pypmc.density.gauss.LocalGauss(np.eye(dim) * 1e-3)
mcs = [pypmc.sampler.markov_chain.AdaptiveMarkovChain(log_target, mc_prop, start) for start in starts]

print('running Markov chains ...')

# In general we need some burn-in.
# Note: In this example, we are already converged into a region of interes.
#       In general, the target is not an indicator and therefore just small
#       outside regions of interest but not zero. The burn-in would move the
#       chain into a region with higher function values than its vicinity.
#       For completeness we do the burn-in here although for this specific
#       example it is unneccessary.
for mc in mcs:
    mc.run(100)
    mc.history.clear()

# Now we let the Markov chains map out the Regions of interest.
# Hereby we use the samples to adapt the proposal function.
for mc in mcs:
    for i in range(100):
        mc.run(100)
        mc.adapt()

mc_samples_sorted_by_chain = [mc.history[:] for mc in mcs]
mc_samples = np.vstack(mc_samples_sorted_by_chain)

# Now we to use the Markov chain samples to generate a proposal function
# for importance sampling. This can for example be done with the variational
# Bayes.
# The variational Bayes takes samples and an initial guess of the output.
# For more information about the following call refer to the documentation
# and the example "Grouping by Gelman-Rubin R value"(r-group.py)
long_patches = pypmc.mix_adapt.r_value.make_r_gaussmix(mc_samples_sorted_by_chain)

# Comments on arguments:
#   o mc_samples[::100]    - Markov chains are auto correlated which means
#                           we may thin the samples
#   o W0=np.eye(dim)*1e5  - The resulting covariance matrices can be very
#                           sensitive to W0. It should be chose much larger
#                           than the actual covariances. If it is too small,
#                           W0 will dominate the resulting covariances and
#                           usually lead to very bad results.
vb = pypmc.mix_adapt.variational.GaussianInference(mc_samples[::100], initial_guess=long_patches, W0=np.eye(dim)*1e10)

# When we run the variational Bayes, we want unneccessary components to
# be automatically pruned. The prune parameter sets how many samples
# a component must effectively have to be considered important. The
# value set here proved to be good in our experiments.
vb_prune = 0.5 * len(vb.data) / vb.K

# Run the variational Bayes for at most 1,000 iterations
print('running variational Bayes ...')
vb.run(1000, rel_tol=1e-10, abs_tol=1e-5, prune=vb_prune, verbose=True)

# extract a Gaussian mixture
vbmix = vb.make_mixture()

# Now we can instanciate an importance sampler. We take "DeterministicIS"
# here which allows the combination of different proposal densities.
# We will now draw 1,000 importance samples and use these for a proposal
# update using variational Bayes again.

print('running importance sampling ...')
sampler = pypmc.sampler.importance_sampling.DeterministicIS(log_target, vbmix)
sampler.run(1000)

# The variational bayes allows us, unlike PMC, to include the information
# gained by the Markov chains in subsequent proposal updates. We know that
# we cannot trust the component weights obtained by the chains. That is
# because the mode a chain find rather depends on the initial position than
# on the actual probability mass of the mode. Nevertheless, we can rely
# on the means and covariances. The following lines show how to code that
# into the variational Bayes.

prior_for_proposal_update = vb.posterior2prior()
prior_for_proposal_update.pop('alpha0')
vb2 = pypmc.mix_adapt.variational.GaussianInference(sampler.history[:][:,1:],
                                                    initial_guess=vbmix,
                                                    weights=sampler.history[:][:,0],
                                                    **prior_for_proposal_update)

# Note: This time we leave "prune" at the default value "1" because all
#       pruning should already have been done in the first variational
#       Bayes.
print('running variational Bayes ...')
vb2.run(1000, rel_tol=1e-10, abs_tol=1e-5, verbose=True)

# Now we draw another 10,000 samples with the updated proposal
sampler.proposal = vb2.make_mixture()
print('running importance sampling ...')
sampler.run(10**4)

# The integral can then be estimated from the weights. The error is also
# estimated out of the weights. By the central limit theorem, the integral
# estimator has a gaussian distribution.
weighted_samples = sampler.history[:]
weights = weighted_samples[:,0]

integral_estimator = weights.sum() / len(weights)
integral_uncertainty_estimator = np.sqrt((weights**2).sum() / len(weights) - integral_estimator**2) / np.sqrt(len(weights))

print("estimated  integral =", integral_estimator, '+-', integral_uncertainty_estimator)
print("analytical integral =", target_integral)

# As mentioned in the very beginning, the methods we applied reinterpret
# the target function as unnormalized probability density.
# In addition to the integral, we also get samples distributed according
# to that probability distribution.
try:
    import matplotlib.pyplot as plt
except ImportError:
    print('For plotting "matplotlib" needs to be installed')
    exit(1)

plt.figure()
plt.hist2d(weighted_samples[:,1], weighted_samples[:,2], weights=weights, bins=100, cmap='gray_r')
pypmc.tools.plot_mixture(sampler.proposal)
plt.show()
