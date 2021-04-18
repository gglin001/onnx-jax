import jax.numpy as jnp
from jax import jit

from onnx_jax.handlers.backend_handler import BackendHandler
from onnx_jax.handlers.handler import onnx_op
from onnx_jax.pb_wrapper import OnnxNode


@onnx_op("Sinh")
class Sinh(BackendHandler):
    @classmethod
    def _common(cls, node: OnnxNode, **kwargs):
        @jit
        def _sinh(x):
            return jnp.sinh(x)

        return _sinh

    @classmethod
    def version_9(cls, node, **kwargs):
        return cls._common(node, **kwargs)
