import jax.numpy as jnp
from jax import jit

from onnx_jax.handlers.backend_handler import BackendHandler
from onnx_jax.handlers.handler import onnx_op
from onnx_jax.pb_wrapper import OnnxNode


@onnx_op("Reciprocal")
class Reciprocal(BackendHandler):
    @classmethod
    def _common(cls, node: OnnxNode, **kwargs):
        @jit
        def _reciprocal(x):
            return jnp.reciprocal(x)

        return _reciprocal

    @classmethod
    def version_1(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def version_6(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def version_13(cls, node, **kwargs):
        return cls._common(node, **kwargs)
