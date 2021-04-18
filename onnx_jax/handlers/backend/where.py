import jax.numpy as jnp
from jax import jit

from onnx_jax.handlers.backend_handler import BackendHandler
from onnx_jax.handlers.handler import onnx_op
from onnx_jax.pb_wrapper import OnnxNode


@onnx_op("Where")
class Where(BackendHandler):
    @classmethod
    def _common(cls, node: OnnxNode, **kwargs):
        return onnx_where

    @classmethod
    def version_9(cls, node, **kwargs):
        return cls._common(node, **kwargs)


@jit
def onnx_where(cond, x, y):
    return jnp.where(cond, x, y)
