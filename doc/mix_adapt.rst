Mixture adaptation
==================

.. automodule:: pypmc.mix_adapt

Hierarchical clustering
-----------------------

.. automodule:: pypmc.mix_adapt.hierarchical

Variational Bayes
-----------------

.. automodule:: pypmc.mix_adapt.variational

PMC
---

.. automodule:: pypmc.mix_adapt.pmc
	:exclude-members: gaussian_pmc, student_t_pmc
.. autofunction::  pypmc.mix_adapt.pmc.gaussian_pmc(samples, density, weights=None, latent=None, rb=True, mincount=0, copy=True)
.. autofunction::  pypmc.mix_adapt.pmc.student_t_pmc(samples, density, weights=None, latent=None, rb=True, dof_solver_steps=100, mindof=1e-5, maxdof=1e3, mincount=0, copy=True)

Gelman-Rubin R-value
--------------------

.. automodule:: pypmc.mix_adapt.r_value
