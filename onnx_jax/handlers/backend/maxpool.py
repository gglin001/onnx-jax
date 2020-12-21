import jax.numpy as jnp
from jax import lax

from onnx_jax.handlers.backend_handler import BackendHandler
from onnx_jax.handlers.handler import onnx_op


@onnx_op("MaxPool")
class MaxPool(BackendHandler):

    @classmethod
    def _common(cls, node, inputs, **kwargs):
        # TODO
        pool_type = "MAX" if len(node.outputs) == 1 else "MAX_WITH_ARGMAX"
        return onnx_maxpool(inputs[0], **node.attrs)

    @classmethod
    def version_1(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def version_8(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def version_10(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def version_11(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def version_12(cls, node, **kwargs):
        return cls._common(node, **kwargs)


def onnx_maxpool(x, kernel_shape, pads=None, strides=None):
    """Numpy-backed implementation of ONNX MaxPool op."""
    prefix = (1,) * (x.ndim - len(kernel_shape))
    dims = prefix + tuple(kernel_shape)
    pads = tuple(pads) if pads else [0] * len(kernel_shape)
    strides = (prefix + tuple(strides)) if strides else [1] * len(kernel_shape)
    return [lax.reduce_window(x, -jnp.inf, lax.max, dims, strides, 'VALID')]
