pypmc
=====

An implementation of Population Monte Carlo (PMC), or adaptive
importance, sampling based on the original proposal by Cappé et al. ().

It uses the initialization of the proposal by Beaujean et al. () using
Markov chains. Other improvements include parallelized proposal
updates, tuning of the student-t degree of freedom ...

With pypmc, the evaluation of the target density can be parallelized
using multiple threads or many processors with mpi4py. The code is
designed such that it does not get into the users way; full control
over how the results are output is available and a convenient default
implementation using hdf5

The basic requirement is only numpy. Optional components include
mpi4py (parallelization), h5py (persistent output to disk in HDF5
format), and matplotlib (plotting).

The code is compatible to both python 2.7 and 3.x.
