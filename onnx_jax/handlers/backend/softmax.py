from jax.nn import softmax

from onnx_jax.handlers.backend_handler import BackendHandler
from onnx_jax.handlers.handler import onnx_op


@onnx_op("Softmax")
class Softmax(BackendHandler):

    @classmethod
    def _common(cls, node, inputs, **kwargs):
        return softmax(inputs[0], node.attrs['axis'])

    @classmethod
    def version_1(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def version_11(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def version_13(cls, node, **kwargs):
        return cls._common(node, **kwargs)
