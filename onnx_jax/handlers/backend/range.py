import jax.numpy as jnp

from onnx_jax.handlers.backend_handler import BackendHandler
from onnx_jax.handlers.handler import onnx_op
from onnx_jax.pb_wrapper import OnnxNode


@onnx_op("Range")
class Round(BackendHandler):
    @classmethod
    def _common(cls, node: OnnxNode, **kwargs):
        def _range(start, limit, delta):
            dtype = start.dtype
            return onnx_range(start, limit, delta, dtype)

        return _range

    @classmethod
    def version_11(cls, node, **kwargs):
        return cls._common(node, **kwargs)


# TODO jit
# @jit
def onnx_range(start, limit, delta, dtype):
    return jnp.arange(start, limit, delta, dtype)
