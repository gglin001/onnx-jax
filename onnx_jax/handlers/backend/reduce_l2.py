import jax.numpy as jnp

from onnx_jax.handlers.backend_handler import BackendHandler
from onnx_jax.handlers.handler import onnx_op


@onnx_op("ReduceL2")
class ReduceL2(BackendHandler):

    @classmethod
    def _common(cls, node, inputs, **kwargs):
        return onnx_reduce_l2(*inputs, **node.attrs)

    @classmethod
    def version_1(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def version_11(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def version_13(cls, node, **kwargs):
        return cls._common(node, **kwargs)


def onnx_reduce_l2(data, axes=None, keepdims=1):
    axes = None if not axes else tuple(axes)
    return [jnp.sqrt(jnp.sum(a=jnp.square(data), axis=axes, keepdims=keepdims == 1))]
