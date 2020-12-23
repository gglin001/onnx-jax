import jax.numpy as jnp

from onnx_jax.handlers.backend_handler import BackendHandler
from onnx_jax.handlers.handler import onnx_op


@onnx_op("And")
class Acos(BackendHandler):

    @classmethod
    def _common(cls, node, inputs, **kwargs):
        return [jnp.logical_and(inputs[0], inputs[1])]

    @classmethod
    def version_1(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def version_7(cls, node, **kwargs):
        return cls._common(node, **kwargs)
