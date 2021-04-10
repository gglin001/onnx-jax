import jax.numpy as jnp

from onnx_jax.handlers.backend_handler import BackendHandler
from onnx_jax.handlers.handler import onnx_op


@onnx_op("Tan")
class Tan(BackendHandler):
    @classmethod
    def _common(cls, node, inputs, **kwargs):
        return [jnp.tan(inputs[0])]

    @classmethod
    def version_7(cls, node, **kwargs):
        return cls._common(node, **kwargs)
