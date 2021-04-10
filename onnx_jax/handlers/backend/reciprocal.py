import jax.numpy as jnp

from onnx_jax.handlers.backend_handler import BackendHandler
from onnx_jax.handlers.handler import onnx_op


@onnx_op("Reciprocal")
class Reciprocal(BackendHandler):
    @classmethod
    def _common(cls, node, inputs, **kwargs):
        return [jnp.reciprocal(inputs[0])]

    @classmethod
    def version_6(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def version_13(cls, node, **kwargs):
        return cls._common(node, **kwargs)
