import jax.numpy as jnp

from onnx_jax.handlers.backend_handler import BackendHandler
from onnx_jax.handlers.handler import onnx_op


@onnx_op("Add")
class Add(BackendHandler):
    @classmethod
    def _common(cls, node, inputs, **kwargs):
        return onnx_add(inputs[0], inputs[1], **node.attrs)

    @classmethod
    def version_1(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def version_6(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def version_7(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def version_13(cls, node, **kwargs):
        return cls._common(node, **kwargs)


def onnx_add(a, b, axis=None, broadcast=False):
    if broadcast:
        b_shape = []
        b_shape.extend(a.shape[:axis])
        b_shape.append(a.shape[axis])
        b_shape.extend([1] * len(a.shape[axis + 1 :]))
        b = jnp.reshape(b, b_shape)
    elif len(a.shape) != len(b.shape):
        b_shape = [1] * len(a.shape)
        b_shape[1] = -1
        b = jnp.reshape(b, b_shape)

    return [jnp.add(a, b)]
