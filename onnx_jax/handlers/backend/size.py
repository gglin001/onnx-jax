import jax.numpy as jnp

from onnx_jax.handlers.backend_handler import BackendHandler
from onnx_jax.handlers.handler import onnx_op


@onnx_op("Size")
class Size(BackendHandler):
    @classmethod
    def _common(cls, node, inputs, **kwargs):
        return [jnp.array(jnp.size(inputs[0])).astype(jnp.int64)]

    @classmethod
    def version_1(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def version_13(cls, node, **kwargs):
        return cls._common(node, **kwargs)
