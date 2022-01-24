import logging

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

def depr_warn_verbose(logger_name):
    logger = logging.getLogger(logger_name)
    logger.warn(
            f"The optional argument 'verbose' is deprecated and "
            f"will be removed in the future. Instead, use the log "
            f"level of logging.getLogger({logger_name}) or a parent."
            )

