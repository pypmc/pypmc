'''This example illustrates how to group Markov Chains according to the
Gelman-Rubin R value (see [GR92]_).

'''

import numpy as np
import pypmc

# A Markov Chain can only explore a local mode of the target function.
# The Gelman-Rubin R value can be used to determine whether N chains
# explored the same mode. Pypmc offers a function which groups chains
# with a common R value less than some ``critical_r``.
#
# In this example, we run five Markov Chains initialized in different
# modes and then group those chains together that explored same mode.


# define a proposal
# this defines the same initial proposal for all chains
prop_dof   = 50.
prop_sigma = np.array([[0.1 , 0.  ]
                      ,[0.  , 0.02]])
prop = pypmc.density.student_t.LocalStudentT(prop_sigma, prop_dof)


# define the target; i.e., the function you want to sample from.
# In this case, it is a bimodal Gaussian with well separated modes.
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

# choose initializations for the chains
# Here we place two chains into the mode at [5, 0.01] and three into the mode at [-4,1].
# In such a setup, the chains will only explore the mode where they are initialized.
# Different random numbers are used in each chain.
starts = [np.array([4.999, 0.])] * 2   +   [np.array([-4.0001, 0.999])] * 3

# define the markov chain objects
mcs = [pypmc.sampler.markov_chain.AdaptiveMarkovChain(log_target, prop, start) for start in starts]

# run and discard burn-in
for mc in mcs:
    mc.run(10**2)
    mc.clear()

# run 10,000 steps adapting the proposal every 500 steps
for mc in mcs:
    for i in range(20):
        mc.run(500)
        mc.adapt()

# extract a reference to the samples from all chains
stacked_values = [mc.samples[:] for mc in mcs]

# find the chain groups
# chains 0 and 1 are initialized in the same mode (at [5, 0.01])
# chains 2, 3 and 4 are initialized in the same mode (at [-4, 0])
# expect chain groups:
expected_groups = [[0,1], [2,3,4]]

# R value calculation only needs the means, variances (diagonal
# elements of covariance matrix) and number of samples,
# axis=0 ensures that we get variances separately for each parameter.
# critical_r can be set manually, here the default value is used
found_groups = pypmc.mix_adapt.r_value.r_group([np.mean(chain, axis=0) for chain in stacked_values],
                                               [np.var (chain, axis=0) for chain in stacked_values],
                                               len(stacked_values[0]))

# print the result
print("Expect %s" % expected_groups)
print("Have   %s"    % found_groups)

# Hint: ``stacked_values`` is an example of what `pypmc.mix_adapt.r_value.make_r_gaussmix()` expects as ``data``
result = pypmc.mix_adapt.r_value.make_r_gaussmix(stacked_values)

try:
    import matplotlib.pyplot as plt
except ImportError:
    print('For plotting "matplotlib" needs to be installed')
    exit(1)

pypmc.tools.plot_mixture(result, cmap='jet')
plt.show()
