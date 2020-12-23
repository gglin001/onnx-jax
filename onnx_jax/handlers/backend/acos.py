import jax.numpy as jnp

from onnx_jax.handlers.backend_handler import BackendHandler
from onnx_jax.handlers.handler import onnx_op


@onnx_op("Acos")
class Acos(BackendHandler):

    @classmethod
    def _common(cls, node, inputs, **kwargs):
        return [jnp.arccos(inputs[0])]

    @classmethod
    def version_7(cls, node, **kwargs):
        return cls._common(node, **kwargs)
