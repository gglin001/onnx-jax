import jax.numpy as jnp

from onnx_jax.handlers.backend_handler import BackendHandler
from onnx_jax.handlers.handler import onnx_op
from onnx_jax.pb_wrapper import OnnxNode


@onnx_op("Tile")
class Tile(BackendHandler):
    @classmethod
    def _common(cls, node: OnnxNode, **kwargs):

        return onnx_tile

    @classmethod
    def version_1(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def version_6(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def version_13(cls, node, **kwargs):
        return cls._common(node, **kwargs)


# TODO jit
# @jit
def onnx_tile(a, b):
    return jnp.tile(a, b)
