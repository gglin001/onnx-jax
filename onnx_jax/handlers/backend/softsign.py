from jax.nn import soft_sign

from onnx_jax.handlers.backend_handler import BackendHandler
from onnx_jax.handlers.handler import onnx_op


@onnx_op("Softsign")
class Softsign(BackendHandler):
    @classmethod
    def _common(cls, node, inputs, **kwargs):
        return soft_sign(inputs[0])

    @classmethod
    def version_1(cls, node, **kwargs):
        return cls._common(node, **kwargs)
