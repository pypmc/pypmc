Sampler
=======

.. automodule:: pypmc.sampler

Markov Chain
------------

.. automodule:: pypmc.sampler.markov_chain
	:exclude-members: MarkovChain, AdaptiveMarkovChain
.. autoclass::  pypmc.sampler.markov_chain.MarkovChain(target, proposal, start, indicator=None, rng=numpy.random.mtrand)
.. autoclass::  pypmc.sampler.markov_chain.AdaptiveMarkovChain(target, proposal, start, indicator=None, rng=numpy.random.mtrand)

Importance Sampling
-------------------

.. automodule:: pypmc.sampler.importance_sampling
	:exclude-members: ImportanceSampler
.. autoclass::  pypmc.sampler.importance_sampling.ImportanceSampler(target, proposal, indicator=None, prealloc=0, rng=numpy.random.mtrand)
