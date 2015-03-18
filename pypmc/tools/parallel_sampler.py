'''Run sampling algorithms in parallel using mpi4py

'''

from mpi4py import MPI

class MPISampler(object):
    '''An MPI4Py parallelized sampler. Parallelizes any :py:mod:`pypmc.sampler`.

    :param sampler_type:

        A class defined in :py:mod:`pypmc.sampler`; the class of the
        sampler to be run in parallel. Example: ``sampler_type=ImportanceSampler``.

    :param comm:

        ``mpi4py`` communicator; the communicator to be used.

    :param args, kwargs:

        Additional arguments which are passed to the constructor of
        ``sampler_type``.

    '''
    def __init__(self, sampler_type, comm=MPI.COMM_WORLD, *args, **kwargs):
        self.sampler = sampler_type(*args, **kwargs)

        self._comm = comm
        self._rank = comm.Get_rank()
        self._size = comm.Get_size() # emcee uses "comm.Get_size() - 1" here for special master treatment

        # master collects samples and weights from other processes
        self.clear()

    def run(self, N=1, *args, **kwargs):
        '''Call the parallelized sampler's ``run`` method. Each process
        will run for ``N`` iterations. Then, the master process (process with
        ``rank = 0``) collects the samples and weights from all processes and
        stores it into ``self.samples_list`` and ``self.weights_list``.
        Master process:   Return a list of the return values from the workers.
        Other  processes: Return the same as the sequential sampler.

        .. seealso::

            :py:class:`pypmc.tools.History`

        :param N:

            Integer; the number of steps to be passed to the ``run`` method.

        :param args, kwargs:

            Additional arguments which are passed to the ``sampler_type``'s
            run method.


        '''
        individual_return = self.sampler.run(N, *args, **kwargs)

        # all workers send samples and weights to master
        self.samples_list = self._comm.gather(self.sampler.samples, root=0)
        if hasattr(self.sampler, 'weights'):
            self.weights_list = self._comm.gather(self.sampler.weights, root=0)

        # master returns list of worker return values
        master_return = self._comm.gather(individual_return, root=0)

        if self._rank == 0:
            return master_return
        else:
            return individual_return

    def clear(self):
        """Delete the history."""
        self.sampler.clear()
        self.samples_list = self._comm.gather(self.sampler.samples, root=0)
        if hasattr(self.sampler, 'weights'):
            self.weights_list = self._comm.gather(self.sampler.weights, root=0)
        else:
            self.weights_list = None
