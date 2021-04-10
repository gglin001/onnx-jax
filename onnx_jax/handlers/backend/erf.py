from jax import lax

from onnx_jax.handlers.backend_handler import BackendHandler
from onnx_jax.handlers.handler import onnx_op


@onnx_op("Erf")
class Erf(BackendHandler):
    @classmethod
    def _common(cls, node, inputs, **kwargs):
        return [lax.erf(inputs[0])]

    @classmethod
    def version_9(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def version_13(cls, node, **kwargs):
        return cls._common(node, **kwargs)
