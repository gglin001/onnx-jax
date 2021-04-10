import jax.numpy as jnp

from onnx_jax.handlers.backend_handler import BackendHandler
from onnx_jax.handlers.handler import onnx_op


@onnx_op("Sinh")
class Sinh(BackendHandler):
    @classmethod
    def _common(cls, node, inputs, **kwargs):
        return [jnp.sinh(inputs[0])]

    @classmethod
    def version_9(cls, node, **kwargs):
        return cls._common(node, **kwargs)
