import inspect
from functools import partial

import jax.numpy as jnp
from jax import jit

from onnx_jax.handlers.backend_handler import BackendHandler
from onnx_jax.handlers.handler import onnx_op
from onnx_jax.pb_wrapper import OnnxNode


@onnx_op("GatherElements")
class GatherElements(BackendHandler):
    @classmethod
    def _common(cls, node: OnnxNode, **kwargs):
        cls._rewrite(node)
        cls._prepare(node)

        return onnx_gather_elements

    @classmethod
    def version_11(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def version_13(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def _rewrite(cls, node: OnnxNode):
        if 'axis' not in node.attrs:
            node.attrs['axis'] = 0

    @classmethod
    def _prepare(cls, node: OnnxNode):
        args = list(inspect.signature(onnx_gather_elements).parameters.keys())
        attrs = [node.attrs.get(k, None) for k in args[node.len_inputs :]]
        node.attrs_list.extend(attrs)


# from https://stackoverflow.com/a/46204790/11767360
@partial(jit, static_argnums=(2))
def onnx_gather_elements(x, indices, axis=0):
    data_swaped = jnp.swapaxes(x, 0, axis)
    index_swaped = jnp.swapaxes(indices, 0, axis)
    gathered = jnp.choose(index_swaped, data_swaped, mode='wrap')
    return jnp.swapaxes(gathered, 0, axis)
