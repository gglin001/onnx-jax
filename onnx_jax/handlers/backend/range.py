import jax.numpy as jnp

from onnx_jax.handlers.backend_handler import BackendHandler
from onnx_jax.handlers.handler import onnx_op


@onnx_op("Range")
class Round(BackendHandler):

    @classmethod
    def _common(cls, node, inputs, **kwargs):
        return [jnp.arange(inputs[0], inputs[1], inputs[2], dtype=jnp.float32)]

    @classmethod
    def version_11(cls, node, **kwargs):
        return cls._common(node, **kwargs)
