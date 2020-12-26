import jax.numpy as jnp

from onnx_jax.handlers.backend_handler import BackendHandler
from onnx_jax.handlers.handler import onnx_op


@onnx_op("ReduceSum")
class ReduceSum(BackendHandler):

    @classmethod
    def _common(cls, node, inputs, **kwargs):
        return onnx_reduce_sum(*inputs, **node.attrs)

    @classmethod
    def version_1(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def version_11(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def version_13(cls, node, **kwargs):
        return cls._common(node, **kwargs)


def onnx_reduce_sum(data, axes=None, keepdims=1, noop_with_empty_axes=0):
    if noop_with_empty_axes:
        # TODO
        raise NotImplemented('ReduceSum with para noop_with_empty_axes != 0')
    axes = None if not axes else tuple(axes)
    return [jnp.sum(data, axis=axes, keepdims=keepdims == 1)]
