from onnx import defs

from onnx_jax.handlers.backend import *  # noqa
from onnx_jax.handlers.backend_handler import BackendHandler
from onnx_jax.logger import logger


def get_all_backend_handlers(opset_dict):
    """Get a dict of all backend handler classes.
    e.g. {'domain': {'Abs': Abs handler class}, ...}, }.

    :param opset_dict: A dict of opset. e.g. {'domain': version, ...}
    :return: Dict.
    """
    handlers = {}
    for handler in BackendHandler.__subclasses__():
        handler.check_cls()

        domain = handler.DOMAIN
        version = opset_dict[domain] if domain in opset_dict else 1
        handler.VERSION = version

        since_version = 1
        if defs.has(handler.ONNX_OP, domain=handler.DOMAIN):
            try:
                since_version = defs.get_schema(
                    handler.ONNX_OP,
                    domain=handler.DOMAIN,
                    max_inclusive_version=version,
                ).since_version
            except RuntimeError:
                logger.error(
                    "Fail to get since_version of {} in domain `{}` "
                    "with max_inclusive_version={}. Set to 1.".format(
                        handler.ONNX_OP, handler.DOMAIN, version
                    )
                )
        else:
            logger.error(
                "Unknown op {} in domain `{}`.".format(
                    handler.ONNX_OP, handler.DOMAIN or "ai.onnx"
                )
            )
        handler.SINCE_VERSION = since_version
        handlers.setdefault(domain, {})[handler.ONNX_OP] = handler
    return handlers
