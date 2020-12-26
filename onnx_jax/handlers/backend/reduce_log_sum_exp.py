import jax.numpy as jnp

from onnx_jax.handlers.backend_handler import BackendHandler
from onnx_jax.handlers.handler import onnx_op


@onnx_op("ReduceLogSumExp")
class ReduceLogSumExp(BackendHandler):

    @classmethod
    def _common(cls, node, inputs, **kwargs):
        return onnx_reduce_log_sum_exp(*inputs, **node.attrs)

    @classmethod
    def version_1(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def version_11(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def version_13(cls, node, **kwargs):
        return cls._common(node, **kwargs)


def onnx_reduce_log_sum_exp(data, axes=None, keepdims=1):
    axes = None if not axes else tuple(axes)
    return [jnp.log(jnp.sum(jnp.exp(data), axis=axes, keepdims=keepdims == 1))]
