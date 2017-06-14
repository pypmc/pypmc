'''Provide classes to organize data storage

'''

import numpy as _np

class History(object):
    """Save a history of 1d-arrays.
    Each call to :py:meth:`.append` is counted as a new "run".

    :param dim:

        Integer; the length of 1d-arrays to be saved.

    :param prealloc:

        Integer; indicates for how many points memory is allocated in advance.
        When more memory is needed, it will be allocated on demand.

    Access:

        ``self[run_nr]`` and ``self[run_begin:run_end]`` return *one* array
        that includes the samples for the runs specified (excluding run_end).

        .. warning::
            Index access returns a reference. Modification changes the history.

        .. hint::
            Negative numbers are supported, for example ``self[-1]`` returns
            the latest run.

    Example:
        >>> h = History(2)
        >>> for i in range(2):
        >>>     a = h.append(i+1)
        >>>     a[:] = i+1
        >>> h[0] # first run
        array([[ 1.,  1.]])
        >>> h[1] # second run
        array([[ 2.,  2.],
               [ 2.,  2.]])
        >>> h[:] # entire history
        array([[ 1.,  1.],
               [ 2.,  2.],
               [ 2.,  2.]])
        >>> len(h) # number of runs
        2

    """
#    :var _points:
#
#        numpy array containing all stored 1d-arrays
#
#    :var _slice_for_run_nr:
#
#        list containing start and stop value to extract an individual run
#        from ``_points``
    def __init__(self, dim, prealloc=1):
        self.dim = int(dim)
        assert self.dim == dim, "``dim`` must be an integer"
        self.prealloc = int(prealloc)
        assert self.prealloc == prealloc, "``prealloc`` must be an integer"
        self.clear()

    def __getitem__(self, item):
        if not self._slice_for_run_nr[item]:
            return _np.array(())
        if type(item) == slice:
            if item.step is not None:
                raise NotImplementedError('strided slicing is not supported')
            index0 = self._slice_for_run_nr[item][ 0][0]
            index1 = self._slice_for_run_nr[item][-1][1]
            return self._points[index0 : index1]
        else:
            return self._points[self._slice_for_run_nr[item][0] : self._slice_for_run_nr[item][-1]]

    def __len__(self):
        return len(self._slice_for_run_nr)

    def append(self, new_points_len):
        '''Allocate memory for a new run and return a reference to that memory
        wrapped in an array of size ``(new_points_len, self.dim)``.

        :param new_points_len:

            Integer; the number of points to be stored in the target memory.

        '''
        new_points_len = int(new_points_len)
        assert new_points_len >= 1, "Must at least append one point!"

        # find out start and stop index of the new memory
        try:
            new_points_start = self._slice_for_run_nr[-1][-1]
        except IndexError:
            new_points_start = 0
        new_points_stop  = new_points_start + new_points_len

        # store slice for new_points
        self._slice_for_run_nr.append( (new_points_start , new_points_stop) )

        if self.memleft < new_points_len: #need to allocate new memory
            self.memleft = 0
            #careful: do not use self._points because this may include unused memory
            self._points  = _np.vstack(  (self[:],_np.empty((new_points_len, self.dim)))  )
        else: #have enough memory
            self.memleft -= new_points_len

        # return reference to the new points
        return self._points[new_points_start:new_points_stop]

    def clear(self):
        """Deletes the history"""
        self._points = _np.empty( (self.prealloc,self.dim) )
        self._slice_for_run_nr = []
        self.memleft = self.prealloc
