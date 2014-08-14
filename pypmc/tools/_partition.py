'''Implements the "minimal lexicographic integer partition"

'''

import numpy as _np
from ..density.gauss import Gauss
from ..density.mixture import MixtureDensity

def partition(N, k):
    '''Distribute ``N`` into ``k`` parts such that each part
    takes the value ``N//k`` or ``N//k + 1`` where ``//`` denotes integer
    division; i.e., perform the minimal lexicographic integer partition.

    Example: N = 5, k = 2  -->  return [3, 2]

    '''
    out = [N // k] * k
    remainder = N % k
    for i in range(remainder):
        out[i] += 1
    return out

def patch_data(data, L=100, try_diag=True, verbose=False):
    '''Patch ``data`` (for example Markov chain output) into parts of
    length ``L``. Return a Gaussian mixture where each component gets
    the empirical mean and covariance of one patch.

    :param data:

        Matrix-like array; the points to be patched. Expect ``data[i]``
        as the d-dimensional i-th point.

    :param L:

        Integer; the length of one patch. The last patch will be shorter
        if ``L`` is not a divisor of ``len(data)``.

    :param try_diag:

        Bool; If some patch does not define a proper covariance matrix,
        it cannot define a Gaussian component. ``try_diag`` defines how
        to handle that case:
        If ``True`` (default), the off-diagonal elements are set to zero
        and it is tried to form a Gaussian with that matrix again. If
        that fails as well, the patch is skipped.
        If ``False`` the patch is skipped directly.

    :param verbose:

        Bool; If ``True`` print all status information.

    '''
    # patch data into length L patches
    patches = _np.array([data[patch_start:patch_start + L] for patch_start in range(0, len(data), L)])

    # calculate means and covs
    means   = _np.array([_np.mean(patch,   axis=0) for patch in patches])
    covs    = _np.array([_np.cov (patch, rowvar=0) for patch in patches])

    # form gaussian components
    components = []
    skipped = []
    for i, (mean, cov) in enumerate(zip(means, covs)):
        try:
            this_comp = Gauss(mean, cov)
            components.append(this_comp)
        except _np.linalg.LinAlgError as error1:
            if verbose:
                print("Could not form Gauss from patch %i. Reason: %s" % (i, repr(error1)))
            if try_diag:
                cov = _np.diag(_np.diag(cov))
                try:
                    this_comp = Gauss(mean, cov)
                    components.append(this_comp)
                    if verbose:
                        print('Diagonal covariance attempt succeeded.')
                except _np.linalg.LinAlgError as error2:
                    skipped.append(i)
                    if verbose:
                        print("Diagonal covariance attempt failed. Reason: %s" % repr(error2))
            else: # if not try_diag
                skipped.append(i)

    # print skipped components if any
    if skipped:
        print("WARNING: Could not form Gaussians from: %s" % skipped)

    # create and return mixture
    return MixtureDensity(components)
