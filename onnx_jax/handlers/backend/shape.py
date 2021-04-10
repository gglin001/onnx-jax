import jax.numpy as jnp

from onnx_jax.handlers.backend_handler import BackendHandler
from onnx_jax.handlers.handler import onnx_op


@onnx_op("Shape")
class Shape(BackendHandler):
    @classmethod
    def _common(cls, node, inputs, **kwargs):
        # return type: int32, not int64
        return [jnp.asarray(inputs[0].shape).astype(jnp.int32)]

    @classmethod
    def version_1(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def version_13(cls, node, **kwargs):
        return cls._common(node, **kwargs)
