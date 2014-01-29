'''This example illustrates how to run a Markov Chain using pypmc'''

import matplotlib.pyplot as plt
import numpy as np
import pypmc

# define a proposal
prop_dof   = 50.
prop_sigma = np.array([[0.1 , 0.  ]
                      ,[0.  , 0.02]])
prop = pypmc.markov_chain.proposal.MultivariateStudentT(prop_sigma, prop_dof)

# define the target; i.e., the function you want to sample from.
# In this case, it is a Gaussian with mean "target_mean" and
# covariance "target_sigma".
#
# Note that the target function "log_target" returns the log of the
# unnormalized gaussian density.
target_sigma = np.array([[0.01 , 0.003 ]
                        ,[0.003, 0.0025]])
inv_target_sigma = np.linalg.inv(target_sigma)
target_mean  = np.array([4.3, 1.1])

def unnormalized_log_pdf_gauss(x, mu, inv_sigma):
    diff = x - mu
    return -0.5 * diff.dot(inv_sigma).dot(diff)

log_target = lambda x: unnormalized_log_pdf_gauss(x, target_mean, inv_target_sigma)

# choose a bad initialization
start = np.array([-2., 10.])

# define the markov chain object
mc = pypmc.markov_chain.markov_chain.AdaptiveMarkovChain(log_target, prop, start)

# run burn-in
mc.run(10**4)

# delete burn-in from history
mc.hist.clear()

# run 100,000 steps adapting the proposal every 500 steps
for i in range(200):
    mc.run(500)
    mc.adapt()

# extract a reference to the history of all visited points
accept_count, values = mc.hist[:]
accept_rate = float(accept_count) / len(values)
print("The chain accepted %4.2f%% of the proposed points" % (accept_rate * 100) )

# plot the result
plt.hexbin(values[:,0], values[:,1], gridsize = 40, cmap='gray_r')
plt.show()
