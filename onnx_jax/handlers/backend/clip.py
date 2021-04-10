import jax.numpy as jnp

from onnx_jax.handlers.backend_handler import BackendHandler
from onnx_jax.handlers.handler import onnx_op


@onnx_op("Clip")
class Clip(BackendHandler):
    @classmethod
    def _common(cls, node, inputs, **kwargs):
        return [jnp.clip(inputs[0], inputs[1], inputs[2])]

    @classmethod
    def version_1(cls, node, **kwargs):
        return [jnp.clip(kwargs['inputs'][0], node.attrs['min'], node.attrs['max'])]

    @classmethod
    def version_6(cls, node, **kwargs):
        return [jnp.clip(kwargs['inputs'][0], node.attrs['min'], node.attrs['max'])]

    @classmethod
    def version_11(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def version_12(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def version_13(cls, node, **kwargs):
        return cls._common(node, **kwargs)
