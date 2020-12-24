from jax.nn import sigmoid

from onnx_jax.handlers.backend_handler import BackendHandler
from onnx_jax.handlers.handler import onnx_op


@onnx_op("Sigmoid")
class Sigmoid(BackendHandler):

    @classmethod
    def _common(cls, node, inputs, **kwargs):
        return [sigmoid(inputs[0])]

    @classmethod
    def version_1(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def version_6(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def version_13(cls, node, **kwargs):
        return cls._common(node, **kwargs)
