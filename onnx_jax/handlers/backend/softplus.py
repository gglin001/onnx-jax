from jax.nn import softplus

from onnx_jax.handlers.backend_handler import BackendHandler
from onnx_jax.handlers.handler import onnx_op


@onnx_op("Softplus")
class Softplus(BackendHandler):

    @classmethod
    def _common(cls, node, inputs, **kwargs):
        return softplus(inputs[0])

    @classmethod
    def version_1(cls, node, **kwargs):
        return cls._common(node, **kwargs)
