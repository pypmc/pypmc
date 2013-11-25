pypmc
=====

The code is compatible to both python 2.7 and 3.x.

To run the extensive test suite, you need the nose test runner. The
tests guarantee agreement of vanilla PMC proposal updates with pmclib
[link]. [having same RNG perhaps impossible. Fix the samples, weights,
and q, then update. Remember 'design for testability']

pypmc requires scipy for special functions and numerical optimization.

It includes a basic Metropolis-Hastings sampler [link to Haario] with
support for adaptive multivariate Gaussian and Student-t proposal
functions.