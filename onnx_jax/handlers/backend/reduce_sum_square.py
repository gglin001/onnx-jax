import jax.numpy as jnp

from onnx_jax.handlers.backend_handler import BackendHandler
from onnx_jax.handlers.handler import onnx_op


@onnx_op("ReduceSumSquare")
class ReduceSumSquare(BackendHandler):

    @classmethod
    def _common(cls, node, inputs, **kwargs):
        return onnx_reduce_sum_square(*inputs, **node.attrs)

    @classmethod
    def version_1(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def version_11(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def version_13(cls, node, **kwargs):
        return cls._common(node, **kwargs)


def onnx_reduce_sum_square(data, axes=None, keepdims=1):
    axes = None if not axes else tuple(axes)
    return [jnp.sum(jnp.square(data), axis=axes, keepdims=keepdims == 1)]
