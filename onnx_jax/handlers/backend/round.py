import jax.numpy as jnp

from onnx_jax.handlers.backend_handler import BackendHandler
from onnx_jax.handlers.handler import onnx_op


@onnx_op("Round")
class Round(BackendHandler):

    @classmethod
    def _common(cls, node, inputs, **kwargs):
        return [jnp.round(inputs[0])]

    @classmethod
    def version_11(cls, node, **kwargs):
        return cls._common(node, **kwargs)
