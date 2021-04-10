import jax.numpy as jnp

from onnx_jax.handlers.backend_handler import BackendHandler
from onnx_jax.handlers.handler import onnx_op


@onnx_op("Mean")
class Mean(BackendHandler):
    @classmethod
    def _common(cls, node, inputs, **kwargs):
        if len(inputs) == 1:
            return [jnp.mean(inputs[0])]
        if len(inputs) == 2:
            return [jnp.add(*inputs) / 2]
        else:
            y = jnp.add(inputs[0], inputs[1])
            for idx in range(2, len(inputs)):
                y = jnp.add(y, inputs[idx])
            return [jnp.divide(y, len(inputs))]

    @classmethod
    def version_1(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def version_6(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def version_8(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def version_13(cls, node, **kwargs):
        return cls._common(node, **kwargs)
