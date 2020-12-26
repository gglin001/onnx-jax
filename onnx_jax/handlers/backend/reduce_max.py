import jax.numpy as jnp
# import numpy as np

from onnx_jax.handlers.backend_handler import BackendHandler
from onnx_jax.handlers.handler import onnx_op


@onnx_op("ReduceMax")
class ReduceMax(BackendHandler):

    @classmethod
    def _common(cls, node, inputs, **kwargs):
        return onnx_reduce_max(*inputs, **node.attrs)

    @classmethod
    def version_1(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def version_11(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def version_12(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def version_13(cls, node, **kwargs):
        return cls._common(node, **kwargs)


def onnx_reduce_max(data, axes=None, keepdims=1):
    axes = None if not axes else tuple(axes)
    # return [jnp.asarray(np.maximum.reduce(data, axis=axes, keepdims=keepdims == 1))]
    return [jnp.max(data, axis=axes, keepdims=keepdims == 1)]
