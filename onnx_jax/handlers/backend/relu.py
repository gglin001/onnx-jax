import jax.numpy as jnp

from onnx_jax.handlers.backend_handler import BackendHandler
from onnx_jax.handlers.handler import onnx_op


@onnx_op("Relu")
class Relu(BackendHandler):

    @classmethod
    def _common(cls, node, inputs, **kwargs):
        return [jnp.maximum(inputs[0], 0)]

    @classmethod
    def version_1(cls, node, **kwargs):
        return cls._common(node, kwargs['inputs'])

    @classmethod
    def version_6(cls, node, **kwargs):
        return cls._common(node, kwargs['inputs'])

    @classmethod
    def version_13(cls, node, **kwargs):
        return cls._common(node, kwargs['inputs'])
