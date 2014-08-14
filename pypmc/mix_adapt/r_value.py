'''Functions associated with the Gelman-Rubin R value [GR92]_.

'''

from __future__ import division as _div
import numpy as _np
from ..tools._doc import _add_to_docstring
from ..tools import partition as _part
from ..density.mixture import create_gaussian_mixture as _mkgauss, create_t_mixture as _mkt

_manual_param_n = ''':param n:

        Integer; the number of samples used to determine the estimates
        passed via ``means`` and ``%(var)s``.

    '''
_manual_param_approx = ''':param approx:

        Bool; If False (default), calculate the R value as in [GR92]_.
        If True, neglect the uncertainty induced by the sampling process.

    '''

@_add_to_docstring(_manual_param_approx)
@_add_to_docstring(_manual_param_n %dict(var='variances'))
def r_value(means, variances, n, approx=False):
    '''Calculate the Gelman-Rubin R value (Chapter 2.2 in [GR92]_).
    The R value can be used to quantify convergence of "Iterative
    Simulations" (e.g. Markov Chains) to their limiting (target)
    distribution. An R value "close to one" indicates convergence.

    .. note::

        The R value is defined only for 1D.

    :param means:

        Vector-like array; the mean value estimates.

    :param variances:

        Vector-like array; the variance estimates.

    '''
    # use same variable names as in [GR92]
    # means is \bar{x}_i
    # variances is s_i^2

    means     = _np.asarray(means)
    variances = _np.asarray(variances)

    assert means.ndim == 1, '``means`` must be vector-like'
    assert variances.ndim == 1, '``variances`` must be vector-like'
    assert len(means) == len(variances), \
    'Number of ``means`` (%i) does not match number of ``variances`` (%i)' %( len(means), len(variances) )

    m = len(means)

    x_bar    = _np.average(means)
    B_over_n = ((means - x_bar)**2).sum() / (m - 1)

    W = _np.average(variances)

    # var_estimate is \hat{\sigma}^2
    var_estimate = (n - 1) / n  *  W   +   B_over_n

    if approx:
        return var_estimate / W

    V = var_estimate + B_over_n / m

    # calculate the three terms of \hat{var}(\hat{V}) (equation (4) in [GR92]
    # var_V is \hat{var}(\hat{V})
    tmp_cov_matrix = _np.cov(variances, means)
    # third term
    var_V = _np.cov(variances, means**2)[1,0] - 2. * x_bar * tmp_cov_matrix[1,0]
    var_V *= 2. * (m + 1) * (n - 1) / (m * m * n)
    # second term (merged n in denominator into ``B_over_n``)
    var_V += ((m + 1) / m)**2 * 2. / (m - 1) * B_over_n * B_over_n
    # first term
    var_V += ((n - 1) / n)**2 / m * tmp_cov_matrix[0,0]

    df = 2. * V**2 / var_V

    if df <= 2.:
        return _np.inf

    return V / W * df / (df - 2)

@_add_to_docstring(_manual_param_approx)
@_add_to_docstring(''':param indices:

        Iterable of Integers; calculate R value only for these dimensions.
        Default is all.

    ''')
@_add_to_docstring(_manual_param_n %dict(var='covs'))
def multivariate_r(means, covs, n, indices=None, approx=False):
    '''Calculate the Gelman-Rubin R value (:py:func:`.r_value`) in each dimension
    for the combination of ``n`` samples from ``m`` chains.

    Each chain is summarized by its mean and covariance. Correlations
    are ignored; i.e., only the diagonal elements of the covariance matrices are
    considered. Return an array with the R values.

    :param means:

        Matrix-like array; the mean value estimates

    :param covs:

        3-dimensional array; the covariance estimates

    '''
    means = _np.asarray(means)
    covs  = _np.asarray(covs)

    assert means.ndim == 2, '``means`` must be matrix-like'
    assert covs.ndim == 3, '``covs`` must be 3-dimensional'
    assert len(means) == len(covs), \
    'Number of ``means`` (%i) does not match number of ``covs`` (%i)' %( len(means), len(covs) )
    assert covs.shape[1] == covs.shape[2], \
    '``covs.shape[1]`` (%i) must match ``covs.shape[2]`` (%i)' %( covs.shape[1], covs.shape[2] )
    assert means.shape[1] == covs.shape[1], \
    'Dimensionality of ``means`` (%i) and ``covs`` (%i) does not match' %( means.shape[1], covs.shape[1] )

    dim = means.shape[1]
    if indices is None:
        indices = range(dim)
    out = _np.empty(len(indices))

    for out_index, dim_index in enumerate(indices):
        assert dim_index < dim, 'All ``indices`` must be less than %i' %dim
        out[out_index] = r_value(means[:,dim_index], covs[:,dim_index,dim_index], n, approx)

    return out

@_add_to_docstring(_manual_param_approx)
@_add_to_docstring(''':param indices:

        Iterable of Integers; calculate and check the R value only for
        these dimensions. Default is all.

    ''')
@_add_to_docstring(''':param critical_r:

        Float; group the chains such that their common R value is below
        ``critical_r``.

    ''')
@_add_to_docstring(_manual_param_n %dict(var='covs'))
def r_group(means, covs, n, critical_r=1.5, indices=None, approx=False):
    '''Group chains whose common :py:func:`.r_value` is less than
    ``critical_r`` in every dimension considered.

    .. seealso::

        :py:func:`.multivariate_r`


    :param means:

        Matrix-like array; the mean value estimates

    :param covs:

        3-dimensional array; the covariance estimates

    '''
    assert len(means) == len(covs)
    means = _np.asarray(means)
    covs  = _np.asarray(covs)

    groups = []

    for i in range(len(means)):
        assigned = False
        # try to assign component i to an existing group
        for group in groups:
            r_values = multivariate_r(means[group + [i]], covs[group + [i]], n, indices, approx)
            if (r_values < critical_r).all():
                # add to group if R value small enough
                group.append(i)
                assigned = True
                break
        # if component i has not been added to an existing group case create a new group
        if not assigned:
            groups.append([i])

    return groups

def _make_r_patches(data, K_g, critical_r, indices, approx):
    '''Helper function for :py:func:`.make_r_gaussmix` and
    :py:func:`.make_r_tmix`. Group the ``data`` according to the R value
    and split each group into ``K_g`` patches. Return the patch means
    and covariances. For details see the docstrings of the above mentioned
    functions.

    '''
    def append_components(means, covs, data, partition):
        subdata_start = 0
        subdata_stop  = partition[0]
        for len_subdata in partition:
            subdata = data[subdata_start:subdata_stop]
            means.append( _np.mean(subdata,   axis=0) )
            covs.append ( _np.cov (subdata, rowvar=0) )
            subdata_start += len_subdata
            subdata_stop  += len_subdata

    n = len(data[0])
    for item in data:
        assert len(item) == n, 'Every chain must bring the same number of points.'

    chain_groups = r_group([_np.mean(chain_values, axis  =0) for chain_values in data],
                           [_np.cov (chain_values, rowvar=0) for chain_values in data],
                           n, critical_r, indices, approx)

    long_patches_means = []
    long_patches_covs = []
    for group in chain_groups:
        # we want K_g components from k_g = len(group) chains
        k_g = len(group)
        if K_g >= k_g:
            # find minimal lexicographic integer partition
            n = _part(K_g, k_g)
            for i, chain_index in enumerate(group):
                # need to partition in n[i] parts
                data_full_chain = data[chain_index]
                # find minimal lexicographic integer partition of chain_length into n[i]
                this_patch_lengths = _part(len(data_full_chain), n[i])
                append_components(long_patches_means, long_patches_covs, data_full_chain, this_patch_lengths)
        else:
            # form one long chain and set k_g = 1
            k_g = 1
            # make one large chain
            data_full_chain = _np.vstack([data[i] for i in group])
            # need to partition into K_g parts -- > minimal lexicographic integer partition
            this_patch_lengths = _part(len(data_full_chain), K_g)
            append_components(long_patches_means, long_patches_covs, data_full_chain, this_patch_lengths)

    return long_patches_means, long_patches_covs

@_add_to_docstring(_manual_param_approx)
def make_r_gaussmix(data, K_g=15, critical_r=2., indices=None, approx=False):
    '''Use ``data`` from multiple "Iterative Simulations" (e.g. Markov
    Chains) to form a Gaussian Mixture. This approach refers to the
    "long patches" in [BC13]_.

    The idea is to group chains according to their R-value as in
    :py:func:`.r_group` and form ``K_g`` Gaussian Components per chain
    group. Once the groups are found by :py:func:`.r_group`, the ``data``
    from each chain group is partitioned into ``K_g`` parts (using
    :py:func:`pypmc.tools.partition`). For each of these parts a Gaussian
    with its empirical mean and covariance is created.

    Return a :py:class:`pypmc.density.mixture.MixtureDensity` with
    :py:class:`pypmc.density.gauss.Gauss` components.

    .. seealso::

        :py:func:`.make_r_tmix`

    :param data:

        Iterable of matrix-like arrays; the individual items are interpreted
        as points from an individual chain.

        .. important::
            Every chain must bring the same number of points.

    :param K_g:

        Integer; the number of components per chain group.

    :param critical_r:

        Float; the maximum R value a chain group may have.

    :param indices:

        Integer; Iterable of Integers; use R value in these dimensions
        only. Default is all.

    '''
    return _mkgauss(*_make_r_patches(data, K_g, critical_r, indices, approx))

@_add_to_docstring(_manual_param_approx)
def make_r_tmix(data, K_g=15, critical_r=2., dof=5., indices=None, approx=False):
    '''Use ``data`` from multiple "Iterative Simulations" (e.g. Markov
    Chains) to form a Student t Mixture. This approach refers to the
    "long patches" in [BC13]_.

    The idea is to group chains according to their R-value as in
    :py:func:`.r_group` and form ``K_g`` Student t Components per chain
    group. Once the groups are found by :py:func:`.r_group`, the ``data``
    from each chain group is partitioned into ``K_g`` parts (using
    :py:func:`pypmc.tools.partition`). For each of these parts a Student t
    component with its empirical mean, covariance and degree of freedom
    is created.

    Return a :py:class:`pypmc.density.mixture.MixtureDensity` with
    :py:class:`pypmc.density.student_t.StudentT` components.

    .. seealso::

        :py:func:`.make_r_gaussmix`

    :param data:

        Iterable of matrix-like arrays; the individual items are interpreted
        as points from an individual chain.

        .. important::
            Every chain must bring the same number of points.

    :param K_g:

        Integer; the number of components per chain group.

    :param critical_r:

        Float; the maximum R value a chain group may have.

    :param dof:

        Float; the degree of freedom the components will have.

    :param indices:

        Integer; Iterable of Integers; use R value in these dimensions
        only. Default is all.

    '''
    assert dof > 2., "``dof`` must be larger than 2. (got %g)" %dof

    means, covs = _make_r_patches(data, K_g, critical_r, indices, approx)

    sigmas  = _np.asarray(covs)
    # cov = nu / (nu-2) * sigma
    sigmas *= (dof - 2.) / dof

    return _mkt(means, sigmas, [dof] * len(means))
