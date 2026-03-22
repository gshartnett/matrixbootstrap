"""
Deprecated: use the standard logging module instead.

    import logging
    logger = logging.getLogger(__name__)

To enable debug output for the matrixbootstrap package, configure the logger
in your application:

    logging.basicConfig(level=logging.DEBUG)
"""
import logging

logger = logging.getLogger("matrixbootstrap")


def debug(s):
    logger.debug(s)
