def merge_function_with_indicator(function, indicator, alternative):
    '''Returns a function such that a call to it is equivalent to:

    if indicator(x):
        return function(x)
    else:
        return alternative

    Note that ``function`` is not called if indicator evaluates to False.


    :param function:

        The function to be called when indicator returns True.

    :param indicator:

        Bool-returning function; the indicator

    :param alternative:

        The object to be returned when indicator returns False

    '''
    if indicator is None:
        return function
    else:
        def merged_function(x):
            if indicator(x):
                return function(x)
            else:
                return alternative
        return merged_function
