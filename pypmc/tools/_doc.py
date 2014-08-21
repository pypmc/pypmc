'''Functions to modify docstrings

'''

def _inherit_docstring(from_class):
    '''Private wrapper function to inherit docstrings from Base class

    Usage example:

    class mood(object):
        'describes typical reactions of a person in a specific mood'
        def how_are_you(self):
            'returns a typical answer to How are you? being in specific mood'
            raise NotImplementedError('No mood specified')

    class good_mood(mood):
        @_inherit_docstring(mood)
        def how_are_you(self):
            print 'Fine, thanks.'



    >>> help(good_mood.how_are_you)

    Help on method how_are_you in module __main__:

    how_are_you(self) unbound __main__.good_mood method
        returns a typical answer to How are you? being in specific mood

    '''
    def wrapper(method):
        funcname = method.__name__
        parent_doc = from_class.__dict__[funcname].__doc__
        if method.__doc__ is not None:
            method.__doc__ += '\n        ' + parent_doc
        else:
            method.__doc__ = parent_doc
        return method
    return wrapper

def _add_to_docstring(string):
    '''Private wrapper function. Appends ``string`` to the
    docstring of the wrapped function.

    '''
    def wrapper(method):
        if method.__doc__ is not None:
            method.__doc__ += string
        else:
            method.__doc__ = string
        return method
    return wrapper
