import jax.numpy as jnp

from onnx_jax.handlers.backend_handler import BackendHandler
from onnx_jax.handlers.handler import onnx_op


@onnx_op("GatherElements")
class GatherElements(BackendHandler):

    @classmethod
    def _common(cls, node, inputs, **kwargs):
        return onnx_gather_elements(*inputs, **node.attrs)

    @classmethod
    def version_11(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def version_13(cls, node, **kwargs):
        return cls._common(node, **kwargs)


# from https://stackoverflow.com/a/46204790/11767360
def onnx_gather_elements(data, indices, axis=0):
    data_swaped = jnp.swapaxes(data, 0, axis)
    index_swaped = jnp.swapaxes(indices, 0, axis)
    gathered = jnp.choose(index_swaped, data_swaped, mode='wrap')
    return [jnp.swapaxes(gathered, 0, axis)]
