'''Implements in PY(thon) the P(opulation)M(onte)C(arlo)

'''

# set the version number
from ._version import __version__

# import these submodules by default
from . import mix_adapt, density, sampler, tools

_log_to_stdout = False
def log_to_stdout(verbose=False):
    '''
    Turn on logging and add a handler which prints to stderr, if not active
    yet.

    :param verbose:

        Bool; if ``True``, output non-critical status information

    '''
    global _log_to_stdout
    import logging
    import sys
    logger = logging.getLogger(__name__)

    if verbose:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.WARNING)

    if not _log_to_stdout:
        formatter = logging.Formatter('[%(levelname)s] %(message)s')
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        _log_to_stdout = True

log_to_stdout()
