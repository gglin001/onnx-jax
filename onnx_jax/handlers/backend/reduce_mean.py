import jax.numpy as jnp

from onnx_jax.handlers.backend_handler import BackendHandler
from onnx_jax.handlers.handler import onnx_op


@onnx_op("ReduceMean")
class ReduceMean(BackendHandler):
    @classmethod
    def _common(cls, node, inputs, **kwargs):
        return onnx_reduce_mean(*inputs, **node.attrs)

    @classmethod
    def version_1(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def version_11(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def version_13(cls, node, **kwargs):
        return cls._common(node, **kwargs)


def onnx_reduce_mean(data, axes=None, keepdims=1):
    axes = None if not axes else tuple(axes)
    # return [jnp.asarray(np.mean(data, axis=axes, keepdims=keepdims == 1))]
    return [jnp.mean(data, axis=axes, keepdims=keepdims == 1)]
