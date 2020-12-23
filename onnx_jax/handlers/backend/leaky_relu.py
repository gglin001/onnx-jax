from jax.nn import leaky_relu

from onnx_jax.handlers.backend_handler import BackendHandler
from onnx_jax.handlers.handler import onnx_op


@onnx_op("LeakyRelu")
class Identity(BackendHandler):

    @classmethod
    def _common(cls, node, inputs, **kwargs):
        return [leaky_relu(inputs[0], node.attrs['alpha'])]

    @classmethod
    def version_1(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def version_6(cls, node, **kwargs):
        return cls._common(node, **kwargs)
