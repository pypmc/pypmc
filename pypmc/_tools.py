'''Private helper functions for general purposes

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
        @inherit_docstring(mood)
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
        method.__doc__ = parent_doc
        return method
    return wrapper
