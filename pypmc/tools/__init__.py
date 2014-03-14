'''Helper functions for general purposes

'''

from ._history import History
# can't plot w/o matplotlib but plotting is
# not an essential feature
try:
    from ._plot import plot_mixture
except ImportError as e:
    pass
from . import indicator
