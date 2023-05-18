
import logging

# ===============
# Runtime Information
# ===============

# ------------------------------------------
# logger
# ------------------------------------------


def create_logger(log_path=None, level='info', cmd_print=True, name=None):
    """
    create  a logging logger

    default log level is 'info'

    Args:
        log_path (str, optional): the path for log file. Defaults to None.
        level (str, optional): log level ('debug' | 'info' | 'warning' | 'error' | 'critical'). Defaults to 'info'.
        cmd_print (bool, optional): whether print log to command line. Defaults to True.
        name (str, optional): logger name. Defaults to None. If no name is specified, return the root logger.

    Returns:
        _type_: _description_
    """
    # input check
    assert log_path or cmd_print, 'log_path and cmd_print cannot be both False'

    # log level mapping
    log_levels = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'critical': logging.CRITICAL
    }

    # log formaters
    formaters = {
        'simple': logging.Formatter('%(message)s'),
        'detailed': logging.Formatter('[%(asctime)s][%(name)s][%(levelname)s] - %(message)s')
    }

    # create a logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # config command log handler
    if cmd_print:
        ch = logging.StreamHandler()
        ch.setLevel(log_levels[level])
        ch.setFormatter(formaters['simple'])
        logger.addHandler(ch)

    # config file log handler
    if log_path:
        fh = logging.FileHandler(log_path)
        fh.setLevel(log_levels[level])
        fh.setFormatter(formaters['detailed'])
        logger.addHandler(fh)

    return logger
