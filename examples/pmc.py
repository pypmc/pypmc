'''This example shows how to use importance sampling and how to
adapt the proposal density using the pmc algorithm.

'''

import numpy as np
import pypmc


# define the target; i.e., the function you want to sample from.
# In this case, it is a bimodal Gaussian
#
# Note that the target function "log_target" returns the log of the
# target function.
component_weights = np.array([0.3, 0.7])

mean0       = np.array ([ 5.0  , 0.01  ])
covariance0 = np.array([[ 0.01 , 0.003 ],
                        [ 0.003, 0.0025]])
inv_covariance0 = np.linalg.inv(covariance0)

mean1       = np.array ([-4.0  , 1.0   ])
covariance1 = np.array([[ 0.1  , 0.    ],
                        [ 0.   , 0.02  ]])
inv_covariance1 = np.linalg.inv(covariance1)

component_means = [mean0, mean1]
component_covariances = [covariance0, covariance1]

target_mixture = pypmc.density.mixture.create_gaussian_mixture(component_means, component_covariances, component_weights)

log_target = target_mixture.evaluate


# define the initial proposal density
# In this case a three-modal gaussian used
# the initial covariances are set to the unit-matrix
# the initial component weights are set equal
initial_prop_means = []
initial_prop_means.append( np.array([ 4.0, 0.0]) )
initial_prop_means.append( np.array([-5.0, 0.0]) )
initial_prop_means.append( np.array([ 0.0, 0.0]) )
initial_prop_covariance = np.eye(2)

initial_prop_components = []
for i in range(3):
    initial_prop_components.append(pypmc.density.gauss.Gauss(initial_prop_means[i], initial_prop_covariance))

initial_proposal = pypmc.density.mixture.MixtureDensity(initial_prop_components)


# define an ImportanceSampler object
sampler = pypmc.sampler.importance_sampling.ImportanceSampler(log_target, initial_proposal)


# draw 10,000 samples adapting the proposal every 1,000 samples
# hereby save the generating proposal component for each sample which is
# returned by sampler.run
# Note: With too few samples components may die out, and one mode might be lost.
generating_components = []
for i in range(10):
    print("\rstep", i, "...\n\t", end='')

    # draw 1,000 samples and save the generating component
    generating_components.append(sampler.run(10**3, trace_sort=True))

    # get a reference to the weights and samples that have just been generated
    samples = sampler.samples[-1]
    weights = sampler.weights[-1][:,0]

    # update the proposal using the pmc algorithm in the non Rao-Blackwellized form
    pypmc.mix_adapt.pmc.gaussian_pmc(samples, sampler.proposal, weights, generating_components[-1],
                                     mincount=20, rb=True, copy=False)

print("\rsampling finished")
print(  '-----------------')
print('\n')

# print information about the adapted proposal
print('initial component weights:', initial_proposal.weights)
print('final   component weights:', sampler.proposal.weights)
print('target  component weights:', component_weights)
print()
for k, m in enumerate([mean0, mean1, None]):
    print('initial mean of component %i:' %k, initial_proposal.components[k].mu)
    print('final   mean of component %i:' %k, sampler.proposal.components[k].mu)
    print('target  mean of component %i:' %k, m)
    print()
print()
for k, c in enumerate([covariance0, covariance1, None]):
    print('initial covariance of component %i:\n' %k, initial_proposal.components[k].sigma, sep='')
    print()
    print('final   covariance of component %i:\n' %k, sampler.proposal.components[k].sigma, sep='')
    print()
    print('target  covariance of component %i:\n' %k, c, sep='')
    print('\n')


# plot results
try:
    import matplotlib.pyplot as plt
except ImportError:
    print('For plotting "matplotlib" needs to be installed')
    exit(1)

def set_axlimits():
    plt.xlim(-6.0, +6.000)
    plt.ylim(-0.2, +1.401)

plt.subplot(221)
plt.title('target mixture')
pypmc.tools.plot_mixture(target_mixture, cmap='jet')
set_axlimits()

plt.subplot(222)
plt.title('pmc fit')
pypmc.tools.plot_mixture(sampler.proposal, cmap='nipy_spectral', cutoff=0.01)
set_axlimits()

plt.subplot(223)
plt.title('target mixture and pmc fit')
pypmc.tools.plot_mixture(target_mixture, cmap='jet')
pypmc.tools.plot_mixture(sampler.proposal, cmap='nipy_spectral', cutoff=0.01)
set_axlimits()

plt.subplot(224)
plt.title('weighted samples')
plt.hist2d(sampler.samples[-1][:,0], sampler.samples[-1][:,1], weights=sampler.weights[-1][:,0], cmap='gray_r', bins=200)
set_axlimits()

plt.tight_layout()
plt.show()
