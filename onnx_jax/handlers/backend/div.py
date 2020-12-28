import jax.numpy as jnp

from onnx_jax.handlers.backend_handler import BackendHandler
from onnx_jax.handlers.handler import onnx_op


@onnx_op("Div")
class Div(BackendHandler):

    @classmethod
    def _common(cls, node, inputs, **kwargs):
        return onnx_div(*inputs, **node.attrs)

    @classmethod
    def version_1(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def version_6(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def version_7(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def version_13(cls, node, **kwargs):
        return cls._common(node, **kwargs)


def onnx_div(a, b, **kwargs):
    return [jnp.divide(a, b)]
