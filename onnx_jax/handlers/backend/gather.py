import jax.numpy as jnp
import numpy as np

from onnx_jax.handlers.backend_handler import BackendHandler
from onnx_jax.handlers.handler import onnx_op


@onnx_op("Gather")
class Gather(BackendHandler):
    @classmethod
    def _common(cls, node, inputs, **kwargs):
        return onnx_gather(*inputs, **node.attrs)

    @classmethod
    def version_1(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def version_11(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def version_13(cls, node, **kwargs):
        return cls._common(node, **kwargs)


def onnx_gather(data, indices, axis=0):
    # if indices has negative number, jnp.take diffs from np.take
    if np.min(indices) < 0:
        return [jnp.asarray(np.take(data, indices, axis=axis))]
    return [jnp.take(data, indices, axis=axis)]
