import jax.numpy as jnp
from jax import jit

from onnx_jax.handlers.backend_handler import BackendHandler
from onnx_jax.handlers.handler import onnx_op
from onnx_jax.pb_wrapper import OnnxNode


@onnx_op("LessOrEqual")
class LessOrEqual(BackendHandler):
    @classmethod
    def _common(cls, node: OnnxNode, **kwargs):
        @jit
        def _less_equal(a, b):
            return jnp.less_equal(a, b)

        return _less_equal

    @classmethod
    def version_12(cls, node, **kwargs):
        return cls._common(node, **kwargs)
