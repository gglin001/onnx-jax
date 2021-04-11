from jax import jit, lax
from onnx_jax.handlers.backend_handler import BackendHandler
from onnx_jax.handlers.handler import onnx_op
from onnx_jax.pb_wrapper import OnnxNode
import jax.numpy as jnp


@onnx_op("And")
class Acos(BackendHandler):
    @classmethod
    def _common(cls, node, inputs, **kwargs):
        @jit
        def _and(a, b):
            return jnp.logical_and(a, b)

        return _and

    @classmethod
    def version_1(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def version_7(cls, node, **kwargs):
        return cls._common(node, **kwargs)
