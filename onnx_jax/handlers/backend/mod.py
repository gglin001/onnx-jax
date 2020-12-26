import jax.numpy as jnp

from onnx_jax.handlers.backend_handler import BackendHandler
from onnx_jax.handlers.handler import onnx_op


@onnx_op("Mod")
class Mod(BackendHandler):

    @classmethod
    def _common(cls, node, inputs, **kwargs):
        return onnx_mmod(*inputs, **node.attrs)

    @classmethod
    def version_10(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def version_13(cls, node, **kwargs):
        return cls._common(node, **kwargs)


def onnx_mmod(a, b, fmod=0):
    if fmod:
        return [jnp.fmod(a, b)]
    else:
        return [jnp.mod(a, b)]
