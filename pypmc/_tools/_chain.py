'''Provides classes to organize data storage

'''

import numpy as _np
from ._doc import _add_to_docstring

class _Chain(object):
    """Abstract base class implementing a sequence of points

    """
    def __init__(self, start):
        self.current = _np.array(start)    # call array constructor to make sure to have a copy
        self.hist    = _hist(self.current) # initialize history

    def run(self, N = 1):
        '''Runs the chain and stores the history of visited points into
        the member variable ``self.hist``

        :param N:

            An int which defines the number of steps to run the chain.

        '''
        raise NotImplementedError()

    def clear(self):
        """Deletes the history

        """
        self.hist = _hist(self.current)

_hist_get_functions_common_part_of_docstring =''':param run_nr:

            int, the number of the run to be extracted

                .. hint::
                    negative numbers mean backwards counting, i.e. the standard
                    value -1 resturns the last run, -2 the run before the last
                    run and so on.

'''

class _hist(object):
    '''Manage history of _Chain objects

    :var points:

        a numpy array containing all visited points in the order of visit

    :var slice_for_run_nr:

        a list containing start and stop value to extract an individual run
        from points

    :var accept_counts:

        a list containing the number of accepted steps in each run

    :param initial_point:

        numpy arrray, the initial point of the chain

    '''
    def __init__(self, initial_point):
        self.points             = initial_point.copy()
        self.slice_for_run_nr   = [(0,1)]
        self.accept_counts      = [1]

    def append(self, new_points, accept_count):
        '''Append a run to the storage

        :param accept_count:

            int, the number of accepted steps in the run to be appended

        :param new_points:

            numpy array, the points visited during the run to be appended

        '''
        # set start and stop for slice of new_points
        new_points_start = self.slice_for_run_nr[-1][-1]
        self.slice_for_run_nr.append((new_points_start , new_points_start + len(new_points)))

        # append the new points to the array
        self.points = _np.vstack((self.points , new_points))

        # append accept count
        self.accept_counts.append(accept_count)

    @_add_to_docstring(_hist_get_functions_common_part_of_docstring)
    def get_run_points(self, run_nr = -1):
        '''Returns a reference to the points of a specific run

        .. warning::
            This function returns a reference. Modification of this functions
            output without explicitly copying it first may result in an
            inconsistent history of the chain!

        '''
        requested_slice = self.slice_for_run_nr[run_nr]
        return self.points[requested_slice[0] : requested_slice[1]]

    @_add_to_docstring(_hist_get_functions_common_part_of_docstring)
    def get_run_accept_count(self, run_nr = -1):
        '''Returns a reference to the points of a specific run

        '''
        return self.accept_counts[run_nr]
