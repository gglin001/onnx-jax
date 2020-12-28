import jax.numpy as jnp

from onnx_jax.handlers.backend_handler import BackendHandler
from onnx_jax.handlers.handler import onnx_op


@onnx_op("Sum")
class Sum(BackendHandler):

    @classmethod
    def _common(cls, node, inputs, **kwargs):
        return [jnp.add(inputs)]

    @classmethod
    def version_1(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def version_6(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def version_8(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def version_13(cls, node, **kwargs):
        return cls._common(node, **kwargs)
