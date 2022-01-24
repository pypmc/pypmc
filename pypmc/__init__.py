'''Implements in PY(thon) the P(opulation)M(onte)C(arlo)

'''

# set the version number
from ._version import __version__

# import these submodules by default
from . import mix_adapt, density, sampler, tools

# Log to stdout per default. The log handler can be removed to use pypmc
# as a library.
tools.util.log_to_stdout()
