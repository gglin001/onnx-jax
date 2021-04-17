import inspect

import jax.numpy as jnp
from jax import jit

from onnx_jax.handlers.backend_handler import BackendHandler
from onnx_jax.handlers.handler import onnx_op
from onnx_jax.pb_wrapper import OnnxNode


@onnx_op("InstanceNormalization")
class InstanceNormalization(BackendHandler):
    @classmethod
    def _common(cls, node: OnnxNode, **kwargs):
        cls._rewrite(node)
        cls._prepare(node)

        return onnx_instancenorm

    @classmethod
    def version_1(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def version_6(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def _rewrite(cls, node: OnnxNode):
        if 'epsilon' not in node.attrs:
            node.attrs['epsilon'] = 1e-5

    @classmethod
    def _prepare(cls, node: OnnxNode):
        args = list(inspect.signature(onnx_instancenorm).parameters.keys())
        attrs = [node.attrs.get(k, None) for k in args[node.len_inputs :]]
        node.attrs_list.extend(attrs)


@jit
def onnx_instancenorm(x, scale, bias, epsilon=1e-5):
    ndim = x.ndim
    axis = tuple(range(2, ndim))
    mean = jnp.mean(x, axis=axis, keepdims=True)
    var = jnp.var(x, axis=axis, keepdims=True)
    dim_ones = (1,) * (ndim - 2)
    scale = scale.reshape(-1, *dim_ones)
    bias = bias.reshape(-1, *dim_ones)
    return scale * (x - mean) / jnp.sqrt(var + epsilon) + bias
