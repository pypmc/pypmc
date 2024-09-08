import numpy as np
import pypmc
from abc import ABCMeta, abstractmethod
from sklearn.base import BaseEstimator, DensityMixin
from numbers import Integral, Real
from time import time
from sklearn.utils._param_validation import Interval, StrOptions

class GaussianMixture(DensityMixin, BaseEstimator, metaclass=ABCMeta):
    """Gaussian Mixture.

    Representation of a Gaussian mixture model probability distribution.
    This class allows to estimate the parameters of a Gaussian mixture
    distribution.

    Read more in the :ref:`User Guide <gmm>`.

    .. versionadded:: 0.18

    Parameters
    ----------
    n_components : int, default=1
        The number of mixture components.

    covariance_type : {'full'}, default='full'
        String describing the type of covariance parameters to use.
        Must be one of:

        - 'full': each component has its own general covariance matrix.

    tol : float, default=1e-3
        The convergence threshold. EM iterations will stop when the
        lower bound average gain is below this threshold.

    reg_covar : float, default=1e-6
        Non-negative regularization added to the diagonal of covariance.
        Allows to assure that the covariance matrices are all positive.

    max_iter : int, default=100
        The number of EM iterations to perform.

    n_init : int, default=1
        The number of initializations to perform. The best results are kept.

    init_params : {'kmeans', 'k-means++', 'random', 'random_from_data'}, \
    default='kmeans'
        The method used to initialize the weights, the means and the
        precisions.
        String must be one of:

        - 'kmeans' : responsibilities are initialized using kmeans.
        - 'k-means++' : use the k-means++ method to initialize.
        - 'random' : responsibilities are initialized randomly.
        - 'random_from_data' : initial means are randomly selected data points.

        .. versionchanged:: v1.1
            `init_params` now accepts 'random_from_data' and 'k-means++' as
            initialization methods.

    weights_init : array-like of shape (n_components, ), default=None
        The user-provided initial weights.
        If it is None, weights are initialized using the `init_params` method.

    means_init : array-like of shape (n_components, n_features), default=None
        The user-provided initial means,
        If it is None, means are initialized using the `init_params` method.

    precisions_init : array-like, default=None
        The user-provided initial precisions (inverse of the covariance
        matrices).
        If it is None, precisions are initialized using the 'init_params'
        method.
        The shape depends on 'covariance_type'::

            (n_components,)                        if 'spherical',
            (n_features, n_features)               if 'tied',
            (n_components, n_features)             if 'diag',
            (n_components, n_features, n_features) if 'full'

    random_state : int, RandomState instance or None, default=None
        Controls the random seed given to the method chosen to initialize the
        parameters (see `init_params`).
        In addition, it controls the generation of random samples from the
        fitted distribution (see the method `sample`).
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    warm_start : bool, default=False
        If 'warm_start' is True, the solution of the last fitting is used as
        initialization for the next call of fit(). This can speed up
        convergence when fit is called several times on similar problems.
        In that case, 'n_init' is ignored and only a single initialization
        occurs upon the first call.
        See :term:`the Glossary <warm_start>`.

    verbose : int, default=0
        Enable verbose output. If 1 then it prints the current
        initialization and each iteration step. If greater than 1 then
        it prints also the log probability and the time needed
        for each step.

    verbose_interval : int, default=10
        Number of iteration done before the next print.

    Attributes
    ----------
    weights_ : array-like of shape (n_components,)
        The weights of each mixture components.

    means_ : array-like of shape (n_components, n_features)
        The mean of each mixture component.

    covariances_ : array-like
        The covariance of each mixture component.
        The shape depends on `covariance_type`::

            (n_components,)                        if 'spherical',
            (n_features, n_features)               if 'tied',
            (n_components, n_features)             if 'diag',
            (n_components, n_features, n_features) if 'full'

    precisions_ : array-like
        The precision matrices for each component in the mixture. A precision
        matrix is the inverse of a covariance matrix. A covariance matrix is
        symmetric positive definite so the mixture of Gaussian can be
        equivalently parameterized by the precision matrices. Storing the
        precision matrices instead of the covariance matrices makes it more
        efficient to compute the log-likelihood of new samples at test time.
        The shape depends on `covariance_type`::

            (n_components,)                        if 'spherical',
            (n_features, n_features)               if 'tied',
            (n_components, n_features)             if 'diag',
            (n_components, n_features, n_features) if 'full'

    precisions_cholesky_ : array-like
        The cholesky decomposition of the precision matrices of each mixture
        component. A precision matrix is the inverse of a covariance matrix.
        A covariance matrix is symmetric positive definite so the mixture of
        Gaussian can be equivalently parameterized by the precision matrices.
        Storing the precision matrices instead of the covariance matrices makes
        it more efficient to compute the log-likelihood of new samples at test
        time. The shape depends on `covariance_type`::

            (n_components,)                        if 'spherical',
            (n_features, n_features)               if 'tied',
            (n_components, n_features)             if 'diag',
            (n_components, n_features, n_features) if 'full'

    converged_ : bool
        True when convergence of the best fit of EM was reached, False otherwise.

    n_iter_ : int
        Number of step used by the best fit of EM to reach the convergence.

    lower_bound_ : float
        Lower bound value on the log-likelihood (of the training data with
        respect to the model) of the best fit of EM.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    BayesianGaussianMixture : Gaussian mixture model fit with a variational
        inference.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.mixture import GaussianMixture
    >>> X = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])
    >>> gm = GaussianMixture(n_components=2, random_state=0).fit(X)
    >>> gm.means_
    array([[10.,  2.],
           [ 1.,  2.]])
    >>> gm.predict([[0, 0], [12, 3]])
    array([1, 0])
    """

    _parameter_constraints: dict = {
        "n_components": [Interval(Integral, 1, None, closed="left")],
        #"tol": [Interval(Real, 0.0, None, closed="left")],
        #"reg_covar": [Interval(Real, 0.0, None, closed="left")],
        "max_iter": [Interval(Integral, 0, None, closed="left")],
        "n_init": [Interval(Integral, 1, None, closed="left")],
        "random_state": ["random_state"],
        "warm_start": ["boolean"],
        "verbose": ["verbose"],
        #"verbose": ["verbose"],
        #"verbose_interval": [Interval(Integral, 1, None, closed="left")],
        "covariance_type": [StrOptions({"full"})],
        "weights_init": ["array-like"],
        "means_init": ["array-like"],
        "precisions_init": ["array-like"],
    }

    def __init__(
        self,
        n_components=1,
        *,
        covariance_type="full",
        tol=1e-3,
        reg_covar=1e-6,
        max_iter=1000,
        n_init=1,
        prune=1.0,
        rel_tol=1e-10,
        abs_tol=1e-5,
        #init_params="kmeans",
        weights_init=None,
        means_init=None,
        covariances_init=None,
        random_state=None,
        warm_start=False,
        verbose=0,
        verbose_interval=10,
    ):
        self.n_components = n_components
        self.tol = tol
        self.reg_covar = reg_covar
        self.max_iter = max_iter
        self.n_init = n_init
        self.prune = prune
        #self.init_params=init_params
        self.random_state = random_state if random_state is not None else np.random
        self.warm_start=warm_start
        self.verbose=verbose
        self.verbose_interval=verbose_interval
        self.covariance_type = covariance_type
        self.weights_init = weights_init
        self.means_init = means_init
        self.covariances_init = covariances_init
        self.rel_tol = rel_tol
        self.abs_tol = abs_tol

        initial_prop_components = [
            pypmc.density.gauss.Gauss(mean, cov)
                for mean, cov in zip(self.means_init, self.covariances_init)]

        self.mix = pypmc.density.mixture.MixtureDensity(initial_prop_components)
        self.state = 'init'

    def fit(self, X, weights=None):
        self.vb = pypmc.mix_adapt.variational.GaussianInference(X, components=len(self.means_init), weights=weights, initial_guess=self.mix)
        self.state = self.vb.run(self.max_iter, self.prune, rel_tol=self.rel_tol, abs_tol=self.abs_tol, verbose=self.verbose>0)
        self.mix = self.vb.make_mixture()

    def score(self, X):
        return self.mix.evaluate(X)
    
    def sample(self, N):
        return self.mix.propose(N, self.random_state)

if __name__ == '__main__':

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
    # define the initial proposal density
    # In this case it has three Gaussians:
    # the initial covariances are set to the unit-matrix,
    # the initial component weights are set equal
    initial_prop_means = []
    initial_prop_means.append( np.array([ 4.0, 0.0]) )
    initial_prop_means.append( np.array([-5.0, 0.0]) )
    initial_prop_means.append( np.array([ 0.0, 0.0]) )
    initial_prop_covariance = np.eye(2)
    initial_prop_covariances = [initial_prop_covariance] * len(initial_prop_means)

    initial_prop_components = [pypmc.density.gauss.Gauss(mean, initial_prop_covariance) for mean in initial_prop_means]
    guess_mix = pypmc.density.mixture.MixtureDensity(initial_prop_components)

    gmix = GaussianMixture(means_init=initial_prop_means, covariances_init=initial_prop_covariances, verbose=2)
    gmix.fit(data)

    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(3, 1, figsize=(5, 12), sharex=True, sharey=True)
    #plt.subplot(311)
    plt.sca(axs[0])
    plt.title('target mixture')
    pypmc.tools.plot_mixture(target_mix, cmap='jet')
    #set_axlimits()

    plt.sca(axs[1])
    plt.title('initial mixture')
    pypmc.tools.plot_mixture(guess_mix, cmap='nipy_spectral', cutoff=0.01)

    plt.sca(axs[2])
    plt.title('variational fit')
    pypmc.tools.plot_mixture(gmix.mix, cmap='nipy_spectral', cutoff=0.01)

    plt.savefig('pypmcMixture.pdf')
