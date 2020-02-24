"""Manage app logging."""
import logging
import os

from config import config

FLASK_ENV = os.getenv('FLASK_ENV')
def logger(name):
    """Return logger for app wide usage."""
    level = "DEBUG" if config[FLASK_ENV].DEBUG else "INFO"
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] [%(filename)s] [method:%(funcName)s]"
        "  --->(%(message)s)")
    logger = logging.getLogger(name)
    return logger
