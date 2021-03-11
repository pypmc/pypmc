===============
Reference Guide
===============


Probability density
===================

.. automodule:: pypmc.density
   :members:
   :show-inheritance:

.. automodule:: pypmc.density.base
   :members:
   :show-inheritance:


Gauss
-----

.. automodule:: pypmc.density.gauss
   :members:
   :show-inheritance:

StudentT
--------

.. automodule:: pypmc.density.student_t
   :members:
   :show-inheritance:

Mixture
-------

.. automodule:: pypmc.density.mixture
   :members:
   :show-inheritance:



Sampler
=======

.. automodule:: pypmc.sampler
   :members:
   :show-inheritance:

Markov Chain
------------

.. automodule:: pypmc.sampler.markov_chain
   :members:
   :show-inheritance:
   :exclude-members: MarkovChain, AdaptiveMarkovChain
.. autoclass::  pypmc.sampler.markov_chain.MarkovChain(target, proposal, start, indicator=None, rng=numpy.random.mtrand)
.. autoclass::  pypmc.sampler.markov_chain.AdaptiveMarkovChain(target, proposal, start, indicator=None, rng=numpy.random.mtrand)

Importance Sampling
-------------------

.. automodule:: pypmc.sampler.importance_sampling
   :members:
   :show-inheritance:
   :exclude-members: ImportanceSampler
.. autoclass::  pypmc.sampler.importance_sampling.ImportanceSampler(target, proposal, indicator=None, prealloc=0, rng=numpy.random.mtrand)



Mixture adaptation
==================

.. automodule:: pypmc.mix_adapt
   :members:
   :show-inheritance:

Hierarchical clustering
-----------------------

.. automodule:: pypmc.mix_adapt.hierarchical
   :members:
   :show-inheritance:

Variational Bayes
-----------------

.. automodule:: pypmc.mix_adapt.variational
   :members:
   :show-inheritance:

PMC
---

.. automodule:: pypmc.mix_adapt.pmc
   :members:
   :show-inheritance:

Gelman-Rubin R-value
--------------------

.. automodule:: pypmc.mix_adapt.r_value
   :members:
   :show-inheritance:



Tools
=====

.. automodule:: pypmc.tools
   :members:
   :show-inheritance:

Convergence diagnostics
-----------------------

.. automodule:: pypmc.tools.convergence
   :members:
   :show-inheritance:

History
-------

.. autoclass:: pypmc.tools.History
   :members:
   :show-inheritance:

Indicator
---------

.. automodule:: pypmc.tools.indicator
.. autofunction:: pypmc.tools.indicator.ball
.. autofunction:: pypmc.tools.indicator.hyperrectangle

.. autofunction:: pypmc.tools.indicator.merge_function_with_indicator

Parallel sampler
----------------

.. automodule:: pypmc.tools.parallel_sampler
   :members:
   :show-inheritance:
   :exclude-members: MPISampler
.. autoclass::  pypmc.tools.parallel_sampler.MPISampler(sampler_type, comm=MPI.COMM_WORLD, mpi_tag=0, *args, **kwargs)

Partition
---------

.. autofunction:: pypmc.tools.partition
.. autofunction:: pypmc.tools.patch_data

Plot
----

.. autofunction:: pypmc.tools.plot_mixture
.. autofunction:: pypmc.tools.plot_responsibility
