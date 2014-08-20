User guide
==========

Mixture density
---------------

Pypmc revolves around adapting mixture densties of the form

.. math::
   q(x) = \sum_{j=1}^K \alpha_j q_j(x), \: \sum_{j=1}^K \alpha_j = 1

where each component :math:`q_j` is either a `Gaussian
<https://en.wikipedia.org/wiki/Normal_distribution>`_

.. math::
   q_j(x) = \mathcal{N}(x | \mu_j, \sigma_j)

or a `student's t <https://en.wikipedia.org/wiki/Student%27s_t-distribution>`_ distribution

.. math::
   q_j(x) = \mathcal{T}(x | \mu_j, \sigma_j, \nu) \,.

The free parameters of the mixture density :math:`\theta` are the component weights
:math:`\alpha_j`, the means :math:`\mu_j`, the covariances
:math:`\sigma_j` and in case :math:`q_j = \mathcal{T}` the degree of
freedom :math:`\nu_j` for :math:`j=1 \dots K`.


Markov chain
------------

PMC
---

Variational Bayes
-----------------

The general idea of variational Bayes is pedagogically explained in
[Bis06]_, Ch. 10. In a nutshell, the unknown joint density of the data
and the hyperparameters is approximated by a distribution that factorizes as

.. math::

   q(X, \vec{\theta}) = q(X) q(\vec{\theta})

In our case, we assume the data to be generated from a mixture of
Gaussians. The priors over the hyperparameters :math:`\vec{\theta}`
are chosen conjugate to the likelihood such that the posterior
:math:`q(\vec{\theta})` has the same functional form as the prior. The
knowledge update due to the data :math:`X` is captured by updating the
values of :math:`\vec{\theta}`. In practice, this results in an
expectation-maximization-like algorithm that converges to a *local*
optimum. It thus depends rather sensitively on the starting values.

We implement two variants of variational Bayes. In either case, one
can specify *all* prior values and the starting points. The *classic*
version is the most well known and widely used. It takes :math:`N`
samples as input. The *mixture reduction* version [BGP10]_ seeks to
compress an input mixture of Gaussians to an output mixture with fewer
components.


Classic version
^^^^^^^^^^^^^^^

A basic example: draw samples from a standard Gaussian in 2D. Then run
variational Bayes to recover that exact Gaussian. Paste the following code into
your python shell and you should get something close to a circle:

.. plot::

   import numpy as np
   import pypmc
   import matplotlib.pyplot as plt

   data = np.random.normal(size=1000).reshape(500, 2)
   vb=pypmc.mix_adapt.variational.GaussianInference(data, components=1)
   vb.run(verbose=True)

   mix=vb.make_mixture()
   pypmc.tools.plot_mixture(mix)
   plt.axes().set_aspect('equal')
   plt.show()

Note that the ``vb`` object carries the posterior distribution of
hyperparameters describing a Gaussan mixture. Invoking
``make_mixture()`` singles out the mixture at the mode of the
posterior. Continuing the example, you can inspect how all
hyperparameters were updated by the data::

   vb.prior_posterior()

and you can check that the mean of the resulting Gaussian is close to
zero and the covariance is close to the identity matrix::

   mix.components[0].mu
   mix.components[0].sigma

In a realistic example, it is usually necessary to give good starting
values to the means of the components in order to accelerate
convergence to a sensible solution. You can pass this information when
you create the ``GaussianInference`` object. Internally, the info is
forwarded to a call to
:meth:`~pypmc.mix_adapt.variational.GaussianInference.set_variational_parameters`,
where all parameter names and symbols are explained in detail.

Mixture reduction
^^^^^^^^^^^^^^^^^


Putting it all together
-----------------------

The examples in the next section show how to use the different algorithms in practice.
