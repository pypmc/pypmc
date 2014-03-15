from __future__ import print_function
import numpy as np
import pypmc

def run(plot=False):
    np.random.seed(12345)

    # define the target; i.e., the function you want to sample from.
    # In this case, it is a bimodal Gaussian

    target_weights = np.array([0.7, 0.3])

    mean0       = np.array ([ 5.0  , 0.01  ])
    covariance0 = np.array([[ 0.01 , 0.003 ],
                            [ 0.003, 0.0025]])

    mean1       = np.array ([-4.0  , 1.0   ])
    covariance1 = np.array([[ 0.1  , 0.    ],
                            [ 0.   , 0.02  ]])

    target_means = [mean0, mean1]
    target_covariances = [covariance0, covariance1]

    # number of samples
    N = 500

    # maximum number of components
    K = 20

    # create the mixture and draw sample data
    mix = pypmc.density.mixture.create_gaussian_mixture(target_means, target_covariances, target_weights)
    data = mix.propose(N=N)
    vb = pypmc.mix_adapt.variational.GaussianInference(data, K)
    vb.run(iterations=100, verbose=True)
    res = vb.make_mixture()

    if plot:
        try:
            import matplotlib.pyplot as plt
            from pypmc.tools._plot import plot_mixture
            plot_mixture(mix, cmap='jet')
            plot_mixture(vb.make_mixture())
            plt.show()
        except ImportError:
            pass

if __name__ == '__main__':
    run()
