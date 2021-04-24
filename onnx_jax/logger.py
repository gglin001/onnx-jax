import logging


def _logger_init():
    logger = logging.getLogger('onnx_jax')
    # do not pass log to the handlers of ancestor loggers
    # https://docs.python.org/3/library/logging.html#logging.Logger.propagate
    logger.propagate = False
    logger.setLevel(logging.INFO)
    stream_hanlder = logging.StreamHandler()
    fmt = "%(name)s - %(created)d %(levelname)s %(filename)s:%(lineno)d %(message)s"
    stream_format = logging.Formatter(fmt)
    stream_hanlder.setFormatter(stream_format)
    logger.addHandler(stream_hanlder)
    return logger


logger = _logger_init()
