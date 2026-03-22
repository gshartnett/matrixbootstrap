import logging

# Best practice for libraries: attach a NullHandler so that log records are
# silently discarded unless the application configures a handler.
logging.getLogger("matrixbootstrap").addHandler(logging.NullHandler())
