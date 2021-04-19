import logging


def _logger_init():
    logger = logging.getLogger('onnx_jax')
    logger.propagate = False
    logger.setLevel(logging.INFO)
    stream_hanlder = logging.StreamHandler()
    fmt = "%(created)d %(levelname)s %(filename)s:%(lineno)d %(message)s"
    stream_format = logging.Formatter(fmt)
    stream_hanlder.setFormatter(stream_format)
    logger.addHandler(stream_hanlder)
    return logger


logger = _logger_init()
