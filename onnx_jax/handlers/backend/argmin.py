import inspect
from functools import partial

import jax.numpy as jnp
from jax import jit

from onnx_jax.handlers.backend_handler import BackendHandler
from onnx_jax.handlers.handler import onnx_op
from onnx_jax.pb_wrapper import OnnxNode


@onnx_op("ArgMin")
class argmin(BackendHandler):
    @classmethod
    def _common(cls, node: OnnxNode, **kwargs):
        cls._rewrite(node)
        cls._prepare(node)

        return onnx_argmax

    @classmethod
    def version_1(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def version_11(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def version_12(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def version_13(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def _rewrite(cls, node: OnnxNode):
        if 'axis' not in node.attrs:
            node.attrs['axis'] = 0
        if 'keepdims' not in node.attrs:
            node.attrs['keepdims'] = 1
        if 'select_last_index' not in node.attrs:
            node.attrs['select_last_index'] = 0

    @classmethod
    def _prepare(cls, node: OnnxNode):
        args = list(inspect.signature(onnx_argmax).parameters.keys())
        attrs = [node.attrs.get(k, None) for k in args[node.len_inputs :]]
        node.attrs_list.extend(attrs)


@partial(jit, static_argnums=(1, 2, 3))
def onnx_argmax(x, axis=0, keepdims=1, select_last_index=0):
    if select_last_index == 0:
        y = jnp.argmin(x, axis=axis)
        if keepdims == 1:
            y = jnp.expand_dims(y, axis)
    else:
        x = jnp.flip(x, axis)
        y = jnp.argmin(x, axis=axis)
        y = x.shape[axis] - y - 1
        if keepdims:
            y = jnp.expand_dims(y, axis)
    return y
