import jax.numpy as jnp

from onnx_jax.handlers.backend_handler import BackendHandler
from onnx_jax.handlers.handler import onnx_op


@onnx_op("Where")
class Where(BackendHandler):

    @classmethod
    def _common(cls, node, inputs, **kwargs):
        return [jnp.where(inputs[0], inputs[1], inputs[2])]

    @classmethod
    def version_9(cls, node, **kwargs):
        return cls._common(node, **kwargs)
