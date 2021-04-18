import jax.numpy as jnp
from jax import jit

from onnx_jax.handlers.backend_handler import BackendHandler
from onnx_jax.handlers.handler import onnx_op
from onnx_jax.pb_wrapper import OnnxNode


@onnx_op("Xor")
class Xor(BackendHandler):
    @classmethod
    def _common(cls, node: OnnxNode, **kwargs):
        return onnx_xor

    @classmethod
    def version_1(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def version_7(cls, node, **kwargs):
        return cls._common(node, **kwargs)


@jit
def onnx_xor(a, b):
    return jnp.logical_xor(a, b)
