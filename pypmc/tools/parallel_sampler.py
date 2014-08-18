'''Run sampling algorithms in parallel using mpi4py

'''

from mpi4py import MPI

class MPISampler(object):
    '''MPISampler(sampler_type, comm=MPI.COMM_WORLD, mpi_tag=0, \*args, \*\*kwargs)

    An MPI4Py parallelized sampler. Parallelizes any :py:mod:`pypmc.sampler`.

    :param sampler_type:

        A class defined in :py:mod:`pypmc.sampler`; the class of the
        sampler to be run in parallel. Example: ``sampler_type=ImportanceSampler``.

    :param comm:

        ``mpi4py`` communicator; the communicator to be used.

    :param mpi_tag:

        Integer; the ``MPISampler`` will only send and receive messages
        with the tag specified here.

        .. important::

            When a method of this sampler is invoked, the ``MPISampler``
            assumes that **ALL MESSAGES TAGGED WITH ``mpi_tag``** are invoked
            by itself.

    :param args, kwargs:

        Additional arguments which are passed to the constructor of
        ``sampler_type``.

    '''
    def __init__(self, sampler_type, comm=MPI.COMM_WORLD, mpi_tag=0, *args, **kwargs):
        self.sampler = sampler_type(*args, **kwargs)

        self._comm = comm
        self._rank = comm.Get_rank()
        self._size = comm.Get_size() # emcee uses "comm.Get_size() - 1" here for special master treatment
        self.mpi_tag = mpi_tag

        # master collects history from other processes
        if self._rank == 0:
            self.history_list = [self.sampler.history]
            for i in range(1, self._size):
                self.history_list.append(self._comm.recv(source=i, tag=self.mpi_tag))
        else:
            self._comm.send(self.sampler.history, tag=self.mpi_tag)

    def run(self, N=1, *args, **kwargs):
        '''Call the parallelized sampler's ``run`` method. Each process
        will run for ``N`` iterations. Then, the master process (process with
        ``self.rank = 0``) collects the data from all processes and stores
        it into ``self.history_list``.
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

        if self._rank == 0:
            master_return = []
            master_return.append(individual_return)

            # reference own history in history_list
            self.history_list = [self.sampler.history]

            # receive history and return from other workers
            for i in range(1, self._size):
                self.history_list.append(self._comm.recv(source=i, tag=self.mpi_tag))
                master_return.append( self._comm.recv(source=i, tag=self.mpi_tag) )

            return master_return

        else: #if self._rank != 0:
            self._comm.send(self.sampler.history, tag=self.mpi_tag)
            self._comm.send(individual_return   , tag=self.mpi_tag)
            return individual_return

    def clear(self):
        """Delete the history."""
        if self._rank == 0:
            for entry in self.history_list:
                entry.clear()

        # special treatment for DeterministicIS --> needs more clearing than just the history
        try:
            self.sampler.clear()
        except AttributeError:
            self.sampler.history.clear()
