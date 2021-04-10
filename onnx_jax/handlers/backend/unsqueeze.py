import jax.numpy as jnp

from onnx_jax.handlers.backend_handler import BackendHandler
from onnx_jax.handlers.handler import onnx_op


@onnx_op("Unsqueeze")
class Unsqueeze(BackendHandler):
    @classmethod
    def _common(cls, node, inputs, **kwargs):
        return onnx_unsqueeze(*inputs, **node.attrs)

    @classmethod
    def version_1(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def version_11(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def version_13(cls, node, **kwargs):
        return cls._common(node, **kwargs)


def onnx_unsqueeze(data, axes=[0], **kwargs):
    for axe in axes:
        data = jnp.expand_dims(data, axis=axe)
    return [data]
