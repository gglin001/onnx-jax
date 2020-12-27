import jax.numpy as jnp

from onnx_jax.handlers.backend_handler import BackendHandler
from onnx_jax.handlers.handler import onnx_op


@onnx_op("Squeeze")
class Squeeze(BackendHandler):

    @classmethod
    def _common(cls, node, inputs, **kwargs):
        return squeeze_impl(*inputs, **node.attrs)

    @classmethod
    def version_1(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def version_11(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def version_13(cls, node, **kwargs):
        return cls._common(node, **kwargs)


def squeeze_impl(data, axes=None, **kwargs):
    if axes is not None:
        axes = tuple(axes)
    return [jnp.squeeze(data, axes)]