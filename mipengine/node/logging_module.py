import logging
import sys
from mipengine.node import config as node_config
from functools import wraps

def getLogger(name):
    logger = logging.getLogger(name)
    formatter = logging.Formatter('%(asctime)s -'
                                  ' %(levelname)s - '
                                  'node - '
                                  '%(name)s - '
                                  '%(funcName)s(%(lineno)d) - '
                                  '%(message)s')
    # logger.setLevel(f"logging.{logLevel}")

    # adding formatting handler to the file handler NOT the logger
    # do it in stdout

    handler = logging.StreamHandler(sys.stdout)
    logger.setLevel(node_config.log_level)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


def logger_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        print(f"*********** Logging for {func.__name__} started ***********")
        output = func(*args, **kwargs)
        print(f"*********** Logging {func.__name__} finished ***********")
        return output
    return wrapper

