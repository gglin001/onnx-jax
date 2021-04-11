import inspect
from functools import partial
from typing import Sequence

import jax.numpy as jnp
from jax import jit

from onnx_jax.handlers.backend_handler import BackendHandler
from onnx_jax.handlers.handler import onnx_op
from onnx_jax.pb_wrapper import OnnxNode


@onnx_op("Compress")
class Compress(BackendHandler):
    @classmethod
    def _common(cls, node: OnnxNode, **kwargs):
        cls._rewrite(node)
        cls._prepare(node)

        def _compress(x, condition, axis=None):
            cond = tuple(list(condition.astype(jnp.int32)))
            return onnx_compress(x, cond, axis)

        return _compress

    @classmethod
    def version_9(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def version_11(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def _rewrite(cls, node: OnnxNode):
        if 'axis' not in node.attrs:
            node.attrs['axis'] = None
        else:
            axis = node.attrs.get('axis')
            if isinstance(axis, Sequence):
                node.attrs['axis'] = tuple(axis)

    @classmethod
    def _prepare(cls, node: OnnxNode):
        args = list(inspect.signature(onnx_compress).parameters.keys())
        attrs = [node.attrs.get(k, None) for k in args[node.len_inputs :]]
        node.attrs_list.extend(attrs)


# TODO jit support
# @partial(jit, static_argnums=(1, 2))
def onnx_compress(x, condition, axis=None):
    y = jnp.compress(condition, x, axis)
    return y
