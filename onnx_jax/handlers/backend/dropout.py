import jax.numpy as jnp

from onnx_jax.handlers.backend_handler import BackendHandler
from onnx_jax.handlers.handler import onnx_op


@onnx_op("Dropout")
class Dropout(BackendHandler):

    @classmethod
    def _common(cls, node, inputs, **kwargs):
        return onnx_dropout(node, *inputs, **node.attrs)

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
    def version_10(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def version_12(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def version_13(cls, node, **kwargs):
        return cls._common(node, **kwargs)


def onnx_dropout(node, data, ratio=0, training_mode=False, seed=None):
    return_mask = len(node.outputs) == 2
    if ratio == 0 or training_mode is False:
        if return_mask:
            return data, jnp.ones(data.shape, dtype=jnp.bool)
        else:
            return [data]
    else:
        # TODO
        raise NotImplemented('Dropout training mode')
