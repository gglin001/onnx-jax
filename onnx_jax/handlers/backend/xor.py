import jax.numpy as jnp

from onnx_jax.handlers.backend_handler import BackendHandler
from onnx_jax.handlers.handler import onnx_op


@onnx_op("Xor")
class Xor(BackendHandler):

    @classmethod
    def _common(cls, node, inputs, **kwargs):
        return [jnp.logical_xor(inputs[0], inputs[1])]

    @classmethod
    def version_1(cls, node, **kwargs):
        return cls.limited_broadcast(node, **kwargs)

    @classmethod
    def version_7(cls, node, **kwargs):
        return [cls.make_tensor_from_onnx_node(node, **kwargs)]
