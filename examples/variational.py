'''This example shows how to generate a "best fit" Gaussian mixture density
from data using variational Bayes.

'''

## in this example, we will:
## 1. Define a Gaussian mixture
## 2. Generate demo data from that Gaussian mixture
## 3. Generate a Gaussian mixture out of the data
## 4. Plot the original and the generated mixture


import numpy as np
import pypmc



# -------------------- 1. Define a Gaussian mixture --------------------

component_weights = np.array([0.3, 0.7])

mean0       = np.array ([ 5.0  , 0.01  ])
covariance0 = np.array([[ 0.01 , 0.003 ],
                        [ 0.003, 0.0025]])

mean1       = np.array ([-4.0  , 1.0   ])
covariance1 = np.array([[ 0.1  , 0.    ],
                        [ 0.   , 0.02  ]])

component_means = [mean0, mean1]
component_covariances = [covariance0, covariance1]

target_mix = pypmc.density.mixture.create_gaussian_mixture(component_means, component_covariances, component_weights)



# -------------------- 2. Generate demo data ---------------------------

data = target_mix.propose(500)



# -------------------- 3. Adapt a Gaussian mixture ---------------------

# maximum number of components
K = 20

# Create a "GaussianInference" object.
# The following command passes just the two essential arguments to "GaussianInference":
# The ``data`` and a maximum number of ``components``.
# For reasonable results in more complicated settings, a careful choice for ``W0``
# is crucial. As a rule of thumb, choose ``inv(W0)`` much smaller than the expected
# covariance. In this case, however, the default (``W0`` = unit matrix) is good enough.
vb = pypmc.mix_adapt.variational.GaussianInference(data, K)

# adapt the variational parameters
converged = vb.run(100, verbose=True)
print('-----------------------------')

# generate a Gaussian mixture with the most probable parameters
fit_mixture = vb.make_mixture()


# -------------------- 4. Plot/print results ---------------------------

if converged is None:
    print('\nThe adaptation did not converge.\n')
else:
    print('\nConverged after %i iterations.\n' %converged)

print("final  component weights: " + str(fit_mixture.weights))
print("target component weights: " + str( target_mix.weights))

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
pypmc.tools.plot_mixture(target_mix, cmap='winter')
set_axlimits()

plt.subplot(222)
plt.title('"best fit"')
pypmc.tools.plot_mixture(fit_mixture, cmap='nipy_spectral')
set_axlimits()

plt.subplot(223)
plt.title('target mixture and "best fit"')
pypmc.tools.plot_mixture(target_mix, cmap='winter')
pypmc.tools.plot_mixture(fit_mixture, cmap='nipy_spectral')
set_axlimits()

plt.subplot(224)
plt.title('data')
plt.hexbin(data[:,0], data[:,1], cmap='gray_r')
set_axlimits()

plt.tight_layout()
plt.show()
