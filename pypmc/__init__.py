'''Implements in PY(thon) the P(opulation)M(onte)C(arlo)

'''

# set the version number
from ._version import __version__

# import these submodules by default
from . import mix_adapt, density, sampler, tools
