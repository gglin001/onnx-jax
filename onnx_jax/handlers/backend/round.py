import jax.numpy as jnp
from jax import jit

from onnx_jax.handlers.backend_handler import BackendHandler
from onnx_jax.handlers.handler import onnx_op
from onnx_jax.pb_wrapper import OnnxNode


@onnx_op("Round")
class Round(BackendHandler):
    @classmethod
    def _common(cls, node: OnnxNode, **kwargs):
        @jit
        def _round(x):
            return jnp.round(x)

        return _round

    @classmethod
    def version_11(cls, node, **kwargs):
        return cls._common(node, **kwargs)
