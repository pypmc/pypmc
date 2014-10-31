Tools
=====

.. automodule:: pypmc.tools

Convergence diagnostics
-----------------------

.. automodule:: pypmc.tools.convergence

History
-------

.. autoclass:: pypmc.tools.History

Indicator
---------

.. automodule:: pypmc.tools.indicator
.. autofunction:: pypmc.tools.indicator.ball
.. autofunction:: pypmc.tools.indicator.hyperrectangle

.. autofunction:: pypmc.tools.indicator.merge_function_with_indicator

Parallel sampler
----------------

.. automodule:: pypmc.tools.parallel_sampler
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
