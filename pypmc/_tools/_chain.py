'''Provides classes to organize data storage

'''

import numpy as _np
from ._doc import _add_to_docstring

class _Chain(object):
    """Abstract base class implementing a sequence of points

    """
    def __init__(self, start, prealloc = 0):
        self.prealloc     =  prealloc
        self.current      = _np.array(start)              # call array constructor to make sure to have a copy
        self.hist         = _Hist(self.current, prealloc) # initialize history

    def run(self, N = 1):
        '''Runs the chain and stores the history of visited points into
        the member variable ``self.hist``

        .. seealso::
            :py:class:`pypmc._tools._chain._Hist`

        :param N:

            An int which defines the number of steps to run the chain.

        '''
        raise NotImplementedError()

_hist_get_functions_common_part_of_docstring =\
''':param run_nr:

            int, the number of the run to be extracted

            .. hint::
                Negative numbers are supported.
                The standard value ``-1`` returns the latest run.\n\n'''

_hist_get_points_functions_warning =\
'''.. warning::\n\
            This function returns a reference. Modification of this\n\
            function's output without explicitly copying it first may\n\
            result in an inconsistent history of the chain!\n\n'''

class _Hist(object):
    """Manages the history of _Chain objects"""
#
#    :var points:
#
#        a numpy array containing all visited points in the order of visit
#
#    :var slice_for_run_nr:
#
#        a list containing start and stop value to extract an individual run
#        from points
#
#    :var accept_counts:
#
#        a list containing the number of accepted steps in each run
#
#    :param initial_point:
#
#        numpy arrray, the initial point of the chain
#
#    :param prealloc:
#
#       int, indicates for how many points memory is allocated in advance
#       When more memory is needed, it will be allocated on demand
#
    def __init__(self, initial_point, prealloc = 0):
        self._dim = len(initial_point)
        if prealloc <= 0:
            self._prealloc       = 0
            self._memleft        = 0
            self._points         = initial_point.copy()
        else:
            self._prealloc       = prealloc
            self._memleft        = prealloc
            self._points         = _np.empty((prealloc + 1, len(initial_point)))
            self._points[0:1]    = initial_point
        self._slice_for_run_nr   = [(0,1)]
        self._accept_counts      = [0]

    def clear(self):
        """Deletes the history"""
        self._memleft        = self._prealloc

        new_first_point      = self.get_run_points()[-1]
        self._points         = _np.empty((self._prealloc + 1, len(new_first_point)))
        self._points[0:1]    = new_first_point

        self._slice_for_run_nr   = [(0,1)]
        self._accept_counts      = [0]

    def _alloc(self, new_points_len):
        '''Allocates memory for a run and returns a reference to that memory

        :param new_points_len:

            int, the number of points to be stored in the target memory

        .. important::

            never call _append_points without a call to _append_accept_count
            otherwise the histoy becomes inconsistent

        '''
        # find out start, stop and len of new_points
        new_points_start = self._slice_for_run_nr[-1][-1]
        new_points_stop  = new_points_start + new_points_len

        # store slice for new_points
        self._slice_for_run_nr.append((new_points_start , new_points_stop))

        if self._memleft < new_points_len: #need to allocate new memory
            self._memleft = 0
            #careful: do not use self._points because this may include unused memory
            self._points  = _np.vstack(( self.get_all_points() , _np.empty((new_points_len, self._dim)) ))

        else: #have enough memory
            self._memleft -= new_points_len

        # return reference to the new points
        return self._points[new_points_start:new_points_stop]

    def _append_accept_count(self, accept_count):
        '''Appends a run's accept count to the storage

        :param accept_count:

            int, the number of accepted steps in the run to be appended

        .. important::

            never call _append_accept_count without a call to _append_points
            otherwise the histoy becomes inconsistent

        '''
        # append accept count
        self._accept_counts.append(accept_count)

    @_add_to_docstring(_hist_get_functions_common_part_of_docstring)
    @_add_to_docstring('        ')
    @_add_to_docstring(_hist_get_points_functions_warning)
    def get_run_points(self, run_nr = -1):
        '''Returns a reference to the points of a specific run

        '''
        requested_slice = self._slice_for_run_nr[run_nr]
        return self._points[requested_slice[0] : requested_slice[1]]

    @_add_to_docstring(_hist_get_functions_common_part_of_docstring)
    def get_run_accept_count(self, run_nr = -1):
        '''Returns the number of accepted steps during specific run

        '''
        return self._accept_counts[run_nr]

    @_add_to_docstring(_hist_get_points_functions_warning)
    def get_all_points(self):
        '''Returns a reference the points visited in all runs including
        the initial point

        '''
        return self._points[:self._slice_for_run_nr[-1][1]]

    def get_all_accept_count(self):
        '''Returns the number of accepted steps during all runs'''
        return sum(self._accept_counts)
