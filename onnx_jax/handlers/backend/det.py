import jax.numpy as jnp
from jax import jit

from onnx_jax.handlers.backend_handler import BackendHandler
from onnx_jax.handlers.handler import onnx_op
from onnx_jax.pb_wrapper import OnnxNode


@onnx_op("Det")
class Det(BackendHandler):
    @classmethod
    def _common(cls, node: OnnxNode, **kwargs):
        @jit
        def _det(x):
            return jnp.linalg.det(x)

        return _det

    @classmethod
    def version_11(cls, node, **kwargs):
        return cls._common(node, **kwargs)
