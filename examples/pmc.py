'''This example shows how to use importance sampling and how to
adapt the proposal density using the pmc algorithm.

'''

from __future__ import print_function
import numpy as np
import pypmc
from pypmc.tools._probability_densities import normalized_pdf_gauss

# define the target; i.e., the function you want to sample from.
# In this case, it is a bimodal Gaussian
#
# Note that the target function "log_target" returns the log of the
# target function and that the target is not normalized.
component_weights = np.array([0.7, 0.3])

mean0       = np.array ([ 5.0  , 0.01  ])
covariance0 = np.array([[ 0.01 , 0.003 ],
                        [ 0.003, 0.0025]])
inv_covariance0 = np.linalg.inv(covariance0)

mean1       = np.array ([-4.0  , 1.0   ])
covariance1 = np.array([[ 0.1  , 0.    ],
                        [ 0.   , 0.02  ]])
inv_covariance1 = np.linalg.inv(covariance1)

component_means = [mean0, mean1, None]
component_covariances = [covariance0, covariance1, None]

def log_target(x):
    target_function_evaluated = component_weights[0] * normalized_pdf_gauss(x, mean0, inv_covariance0) + \
                                component_weights[1] * normalized_pdf_gauss(x, mean1, inv_covariance1)
    # break normalization
    target_function_evaluated *= 10.
    # return the log of the target function; explicitly catch zeros to avoid the call np.log(0)
    if target_function_evaluated == 0.:
        return -np.inf
    else:
        return np.log(target_function_evaluated)


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


# run 100,000 steps adapting the proposal every 10,000 steps
# hereby save the generating proposal component for each sample which is
# returned by mc.run
generating_components = []
for i in range(10):
    print("\rstep", i, "...\n\t", end='')

    # run 10,000 steps and save the generating component
    generating_components.append(sampler.run(10**4, trace_sort=True))

    # get the weighted samples that have just been generated
    weighted_samples = sampler.history[-1]

    # update the proposal using the pmc-algorithm in the non Rao-Blackwellized form
    pypmc.mix_adapt.pmc.gaussian_pmc(weighted_samples, sampler.proposal, generating_components[-1], mincount=20, rb=False, copy=False)

print("\rsampling finished")
print(  '-----------------')
print('\n')

# print information about the adapted proposal
print('initial component weights:', initial_proposal.weights)
print('final   component weights:', sampler.proposal.weights)
print('target  component weights:', component_weights)
print()
for k in range(3):
    print('initial mean of component %i:' %k, initial_proposal.components[k].mu)
    print('final   mean of component %i:' %k, sampler.proposal.components[k].mu)
    print('target  mean of component %i:' %k, component_means[k])
    print()
print()
for k in range(3):
    print('initial covariance of component %i:\n' %k, initial_proposal.components[k].sigma, sep='')
    print()
    print('final   covariance of component %i:\n' %k, sampler.proposal.components[k].sigma, sep='')
    print()
    print('target  covariance of component %i:\n' %k, component_covariances[k], sep='')
    print('\n')
