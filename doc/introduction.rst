Overview
========

Let :math:`P` denote the target target density, and let :math:`q`
equal the proposal density. Then the basic idea of importance sampling
is to approximate the integral of :math:`P` as

.. math::
   \int \mbox{d} x \, P(x) = \int \mbox{d} x \, q(x) \frac{P(x)}{q(x)}
   \approx \frac{1}{N} \sum_{i=1}{N} \frac{P(x_i)}{q(x_i)}

where each :math:`x` is a :math:`D`-dimensional vector drawn
independently from :math:`q`. The :math:`i`-th importance weight is
defined as

.. math::
   w_i \equiv \frac{P(x_i)}{q(x_i)}

The most accurate estimate is obtained for :math:`q=P`, so the goal is
make :math:`q` as close as possible to :math:`P`.

In pypmc, we choose :math:`q` to be a mixture density composed of
either Gaussian or student's t components :math:`q_j`

.. math::
   q(x) = \sum_j \alpha_j q_j(x), \: \sum_j \alpha_j = 1 \,.

Initial proposal density
------------------------

The key ingredient to make adaptive importance sampling work is a good
initial proposal density that closely resembles the target density. A
general method to automatically determine the bulk of the target is to
run multiple Markov chains, and to use clustering to extract a mixture
density from the samples [BC13]_. We provide a generic implementation
of adaptive local-random-walk MCMC [HST01]_ featuring Gauss and
student's t local proposals. MCMC can be used standalone and is
usually all one needs for a unimodal distribution if the evidence is
not of interest. For the clustering, we offer several options. At the
level of individual samples, we have

* population Monte Carlo [Cap+08]_
* variational Bayes for Gaussian mixtures [Bis06]_

and at the level of Gaussian mixtures, there is

* hierarchical clustering [GR04]_ as suggested by Beaujean & Caldwell
  [BC13]_
* variational Bayes (VBmix) [BGP10]_

Proposal updates
----------------

Starting with an initial proposal, samples are drawn from the proposal
:math:`q`, the importance weights are computed, and the proposal is
updated using the samples and weights to more closely approximate the
target density. The two main update algorithms included are:

* Population Monte Carlo (PMC)
* Variational Bayes (VB)

PMC
---

Based on the original proposal by Cappé et al. [Cap+08]_, we offer
updates for a mixture of Gaussian or student's t components. Important
improvements are:

* The option to adapt the student's t degree of freedom - individually
  for each component - as in [HOD12]_. That's one less parameter that
  the user has to guess.
* The power to combine the proposals of subsequent steps
  [Cor+12]_. This increases the effective sample size per wallclock
  time and helps in reducing undesired samples with very large weight --- *outliers* ---
  that adversely affect the variance of the integral estimate.

Variational Bayes
-----------------

A powerful alternative to PMC is to use the variational Bayes
algorithm [Bis06]_ to fit a Gaussian mixture to samples. We include a
variant that also works with importance-weighted samples. Our
implementation allows the user to set all values of the
prior/posterior hyperparameters. Variational Bayes can therefore be
used in a sequential manner to incrementally update the knowledge
about the Gaussian mixture as new (importance) samples arrive.

Performance
-----------

Importance sampling naturally lends itself to massive parallelization
because once the samples are drawn from the proposal, the computation
of N importance weights requires N independent calls to the target
density. Even for moderately complicated problems, these calls are
typically the most expensive part of the calculation. With pypmc, the
importance weights can optionally be computed in multiple processes on
a single machine or a whole cluster with mpi4py. Similarly, multiple
Markov chains are independent and can be run in separate processes.

The second major contribution to overall computing time is the update
algorithm itself. We profiled the program and transferred the relevant
loops from python to compiled C code via cython.

The code is designed such that it does not get into the users way;
full control over how individual components interact is a major design
goal.
