import inspect
from functools import partial

import jax.numpy as jnp
from jax import jit

from onnx_jax.handlers.backend_handler import BackendHandler
from onnx_jax.handlers.handler import onnx_op
from onnx_jax.pb_wrapper import OnnxNode


@onnx_op("ReduceLogSumExp")
class ReduceLogSumExp(BackendHandler):
    @classmethod
    def _common(cls, node: OnnxNode, **kwargs):
        cls._rewrite(node)
        cls._prepare(node)

        return onnx_reduce_log_sum_exp

    @classmethod
    def version_1(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def version_11(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def version_13(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def _rewrite(cls, node: OnnxNode):
        if 'axes' in node.attrs:
            axes = node.attrs['axes']
            node.attrs['axes'] = tuple(axes)
        if 'keepdims' not in node.attrs:
            node.attrs['keepdims'] = 1

    @classmethod
    def _prepare(cls, node: OnnxNode):
        args = list(inspect.signature(onnx_reduce_log_sum_exp).parameters.keys())
        attrs = [node.attrs.get(k, None) for k in args[node.len_inputs :]]
        node.attrs_list.extend(attrs)


@partial(jit, static_argnums=(1, 2))
def onnx_reduce_log_sum_exp(data, axes=None, keepdims=1):
    return jnp.log(jnp.sum(jnp.exp(data), axis=axes, keepdims=keepdims == 1))
