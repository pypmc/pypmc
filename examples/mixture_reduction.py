'''Demonstrate the usage of hierarchical clustering and variational
Bayes (VBMerge) to reduce a given Gaussian mixture to a Gaussian
mixture with a reduced number of components.

'''

import numpy as np
from scipy.stats import chi2
import pypmc

# dimension
D = 2

# number of components
K = 400

# Wishart parameters: mean W, degree of freedom nu
W = np.eye(D)
nu = 5

# "draw" covariance matrices from Wishart distribution
def wishart(nu, W):
    dim = W.shape[0]
    chol = np.linalg.cholesky(W)
    tmp = np.zeros((dim,dim))
    for i in range(dim):
        for j in range(i+1):
            if i == j:
                tmp[i,j] = np.sqrt(chi2.rvs(nu-(i+1)+1))
            else:
                tmp[i,j] = np.random.normal(0,1)
    return np.dot(chol, np.dot(tmp, np.dot(tmp.T, chol.T)))

covariances = [wishart(nu, W) for k in range(K)]

# put components at positions drawn from a Gaussian around mu
mu = np.zeros(D)
means = [np.random.multivariate_normal(mu, sigma) for sigma in covariances]

# equal weights for every component
weights = np.ones(K)

# weights are automatically normalized
input_mixture = pypmc.density.mixture.create_gaussian_mixture(means, covariances, weights)

# create initial guess from first K_out components
K_out = 10
initial_guess = pypmc.density.mixture.create_gaussian_mixture(means[:K_out], covariances[:K_out], weights[:K_out])

###
# hierarchical clustering
#
# - the output closely resembles the initial guess
# - components laid out spherically symmetric
# - every component is preserved
###
h = pypmc.mix_adapt.hierarchical.Hierarchical(input_mixture, initial_guess)
h.run(verbose=True)

###
# VBMerge
#
# - N is the number of samples that gave rise to the input mixture. It
#   is arbitrary, so play around with it. You might have to adjust the
#   ``prune`` parameter in the ``run()`` method
# - only one component survives, again it is spherically symmetric
###
vb = pypmc.mix_adapt.variational.VBMerge(input_mixture, N=1000,
                                         initial_guess=initial_guess)
print()
print("Start variational Bayes:")
vb.run(verbose=True)

# plot results
try:
    import matplotlib.pyplot as plt
except ImportError:
    print('For plotting "matplotlib" needs to be installed')
    exit(1)

def set_axlimits():
    plt.gca().set_aspect('equal')
    plt.xlim(-12.0, +12.0)
    plt.ylim(-12.0, +12.0)

plt.subplot(221)
plt.title('input mixture')
pypmc.tools.plot_mixture(input_mixture)
set_axlimits()

plt.subplot(222)
plt.title('initial guess')
pypmc.tools.plot_mixture(initial_guess)
set_axlimits()

plt.subplot(223)
plt.title('variational Bayes')
pypmc.tools.plot_mixture(vb.make_mixture(), cmap='autumn')
set_axlimits()

plt.subplot(224)
plt.title('hierarchical output')
pypmc.tools.plot_mixture(h.g)
set_axlimits()

plt.tight_layout()
plt.show()
