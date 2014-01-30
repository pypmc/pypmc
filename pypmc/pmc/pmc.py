"""Collect Population Monte Carlo

"""

import numpy as _np
from math import exp
from copy import deepcopy as _cp
from .importance_sampling import ImportanceSampler as _ImportanceSampler
from .._tools._doc import _inherit_docstring, _add_to_docstring
from .._tools._chain import _Hist
from .._tools._regularize import regularize

class GaussianPMC(_ImportanceSampler):
    # set the docstring --> inherit from Base class, but replace:
    # - _ImportanceSampler(*args, **kwargs) --> GaussianPMC(*args, **kwargs)
    # - An importance sampler --> An adaptive importance sampler
    # - add References [Cap+08] and [Cor+12]
    # - proposal density --> intial proposal density and should be Gaussian Mixture
    __doc__ = _ImportanceSampler.__doc__\
    .replace('ImportanceSampler(', 'GaussianPMC(')\
    .replace('An importance sampler', '''An adaptive importance sampler''' , 1)\
    .replace('``proposal``.', '''``proposal``.
    Adaptation uses the (M-)PMC algorithm according to [Cap+08]_ but with
    deterministic mixture weights according to [Cor+12]_.''' , 1)\
    .replace('''The proposal density `q`.
        Should be of type :py:class:`pypmc.pmc.proposal.PmcProposal`.''',
        '''The initial proposal density :math:`q_0`.

        Should be of type :py:class:`pypmc.pmc.proposal.MixtureProposal`.

        Should only have components of type
        :py:class:`pypmc.pmc.proposal.GaussianComponent`.''',1)

    def __init__(self, *args, **kwargs):
        super(GaussianPMC, self).__init__(*args, **kwargs)
        self.proposal_hist = [_cp(self.proposal)]
        self._deltas_targets_evaluated = \
                    _Hist( _np.array([
                    exp(self.proposal.evaluate(self.hist[0][1][0][1:])), # initial delta
                    exp(self.target(self.hist[0][1][0][1:])) # initial target
                    ]), self.hist._prealloc )

    def run(self, N=1):
        """Appends N new points to the history of visited points ``self.hist``
        and recalculates all weights according to [Cor+12]_.

        .. seealso::
            :py:class:`pypmc._tools._chain._Hist`

        :param N:

            An int which defines the number of steps to run the chain.

        """
        super(GaussianPMC, self).run(N)

    @_inherit_docstring(_ImportanceSampler)
    def _calculate_weights(self, this_weights_samples, this_N):
        inconsistency_message = 'Inconsistent state encountered. If you used ' + \
                                '``self.hist.clear()`` try ``self.clear()`` instead.'

        # append proposal for this run to history
        self.proposal_hist.append(_cp(self.proposal))

        # allocate memory for new target and proposal evaluations
        this_deltas_targets = self._deltas_targets_evaluated._alloc(this_N)
        self._deltas_targets_evaluated._append_accept_count(this_N)

        # create references
        this_weights = this_weights_samples[:,0 ]
        this_samples = this_weights_samples[:,1:]
        this_deltas  = this_deltas_targets [:,0 ]
        this_targets = this_deltas_targets [:,1 ]

        num_old_samples1, old_weights_samples = self.hist[:-1]
        num_old_samples2, old_deltas_targets  = self._deltas_targets_evaluated[:-1]
        assert num_old_samples1 == num_old_samples2, inconsistency_message

        old_weights = old_weights_samples[:,0 ]
        old_samples = old_weights_samples[:,1:]
        old_deltas  = old_deltas_targets [:,0 ]
        old_targets = old_deltas_targets [:,1 ]

        num_all_samples1, all_weights_samples = self.hist[:]
        num_all_samples2, all_deltas_targets  = self._deltas_targets_evaluated[:]
        assert num_all_samples1 == num_all_samples2, inconsistency_message

        all_weights = all_weights_samples[:,0 ]
        all_samples = all_weights_samples[:,1:]
        all_deltas  = all_deltas_targets [:,0 ]
        all_targets = all_deltas_targets [:,1 ]


        # evaluate the target at the new samples
        for i, sample in enumerate(this_samples):
            # exp because the self.target returns the log of the target
            this_targets[i] = exp(self.target(sample))

        # calculate the deltas for the new samples
        this_deltas[:] = 0.
        for i_sample, sample in enumerate(this_samples):
            for i_run, (num_samples, dummy) in enumerate(self.hist):
                this_deltas[i_sample] += num_samples * exp( self.proposal_hist[i_run].evaluate(sample) )

        assert i_run + 1 == len(self.proposal_hist), inconsistency_message

        # calculate the deltas for the old samples
        for i_sample, sample in enumerate(old_samples):
            old_deltas[i_sample] += this_N * exp( self.proposal_hist[-1].evaluate(sample) )

        # calculate the weights (Algorithm1 in [Cor+12])
        all_weights[:] = all_targets / (all_deltas / num_all_samples1)

    def adapt(self):
        """Implements the (M-)PMC algorithm according to [Cap+08]_ but uses
        deterministic mixture weights according to [Cor+12]_. These weights
        are calculated along with :py:meth:`run`.

        """
        number_of_samples, weights_samples = self.hist[:]
        normalized_weights = weights_samples[:,0] / weights_samples[:,0].sum()
        samples = weights_samples[:,1:]
        rho = _np.empty(( number_of_samples,len(self.proposal.components) ))
        for k in range(len(self.proposal.components)):
            for n, sample in enumerate(samples):
                rho[n, k]  = exp(self.proposal.components[k].evaluate(sample)) * self.proposal.weights[k]
                # avoid division by zero
                rho[n, k] /= exp(self.proposal.evaluate(sample)) + _np.finfo('d').tiny

        # update equations according to (14) in [Cap+08]
        # ----------------------------------------------

        # new component weights
        alpha = _np.einsum('n,nk->k', normalized_weights, rho)
        regularize(alpha)
        inv_alpha = 1./alpha

        # new means
        mu = _np.einsum('n,nk,nd->kd', normalized_weights, rho, samples)
        mu = _np.einsum('kd,k->kd', mu, inv_alpha)

        # new covars
        cov = _np.empty(( len(mu),len(samples[0]),len(samples[0]) ))
        for k in range(len(self.proposal.components)):
            x_minus_mu = samples - mu[k]
            cov[k] = _np.einsum('n,n,ni,nj->ij', normalized_weights, rho[:,k], x_minus_mu, x_minus_mu) * inv_alpha[k]

        # ----------------------------------------------


        # apply the updated mixture weights, means and covariances
        # --------------------------------------------------------

        for k, component in enumerate(self.proposal.components):
            self.proposal.weights[k] = alpha[k]

            # check if matrix is positive definite:
            # if not, the update will fail
            # in that case replug the old values
            old_mu    = component.mu    # do not need to copy because .update creates a new array
            old_sigma = component.sigma # do not need to copy because .update creates a new array
            try:
                component.update(mu[k], cov[k])
            except _np.linalg.LinAlgError:
                component.update(old_mu, old_sigma)

    def clear(self):
        """Deletes the history"""
        self.hist.clear()
        self._deltas_targets_evaluated.clear()
        self.proposal_hist = [self.proposal_hist[-1]]
