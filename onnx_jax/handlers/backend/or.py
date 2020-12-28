import jax.numpy as jnp

from onnx_jax.handlers.backend_handler import BackendHandler
from onnx_jax.handlers.handler import onnx_op


@onnx_op("Or")
class Or(BackendHandler):

    @classmethod
    def _common(cls, node, inputs, **kwargs):
        return onnx_or(*inputs, **node.attrs)

    @classmethod
    def version_1(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def version_7(cls, node, **kwargs):
        return cls._common(node, **kwargs)


def onnx_or(a, b, **kwargs):
    return[jnp.logical_or(a, b)]
