import jax.numpy as jnp
from jax import jit

from onnx_jax.handlers.backend_handler import BackendHandler
from onnx_jax.handlers.handler import onnx_op
from onnx_jax.pb_wrapper import OnnxNode


@onnx_op("Not")
class Not(BackendHandler):
    @classmethod
    def _common(cls, node: OnnxNode, **kwargs):
        @jit
        def _not(x):
            return jnp.logical_not(x)

        return _not

    @classmethod
    def version_1(cls, node, **kwargs):
        return cls._common(node, **kwargs)
