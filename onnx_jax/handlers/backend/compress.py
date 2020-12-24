import jax.numpy as jnp

from onnx_jax.handlers.backend_handler import BackendHandler
from onnx_jax.handlers.handler import onnx_op


@onnx_op("Compress")
class Compress(BackendHandler):

    @classmethod
    def _common(cls, node, inputs, **kwargs):
        return onnx_compress(*inputs, **node.attrs)

    @classmethod
    def version_9(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def version_11(cls, node, **kwargs):
        return cls._common(node, **kwargs)


def onnx_compress(x, condition, axis=None):
    return [jnp.compress(condition, x, axis)]
