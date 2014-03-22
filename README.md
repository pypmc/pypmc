pypmc
=====

An implementation of Population Monte Carlo (PMC), or adaptive
importance, sampling based on the original proposal by Cappé et
al. ().

It uses the initialization of the proposal by Beaujean et al. () using
Markov chains. Other improvements include parallelized proposal
updates, tuning of the student-t degree of freedom ...

An alternative is to use the variational Bayes algorithm by Bishop ()
to fit a Gaussian mixture to samples. We include a variant that works
with importance-weighted samples.

With pypmc, the evaluation of the target density can be parallelized
using multiple threads or many processors with mpi4py. The code is
designed such that it does not get into the users way; full control
over how individual components interact is a major design goal.

Required packages are numpy (1.6), scipy (0.9), and cython
(0.20). Optional components include mpi4py (parallelization),
matplotlib (plotting), and nose (testing).

The code is compatible to both python 2.7 and 3.x, and has been tested
and developed on Ubuntu 12.04 and 13.10.
