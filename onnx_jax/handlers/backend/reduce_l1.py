import jax.numpy as jnp

from onnx_jax.handlers.backend_handler import BackendHandler
from onnx_jax.handlers.handler import onnx_op


@onnx_op("ReduceL1")
class ReduceL1(BackendHandler):
    @classmethod
    def _common(cls, node, inputs, **kwargs):
        return onnx_reduce_l1(*inputs, **node.attrs)

    @classmethod
    def version_1(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def version_11(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def version_13(cls, node, **kwargs):
        return cls._common(node, **kwargs)


def onnx_reduce_l1(data, axes=None, keepdims=1):
    axes = None if not axes else tuple(axes)
    return [jnp.sum(a=jnp.abs(data), axis=axes, keepdims=keepdims == 1)]
