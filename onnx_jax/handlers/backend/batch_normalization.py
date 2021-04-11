import inspect
from functools import partial

from jax import jit, lax
from onnx_jax.handlers.backend_handler import BackendHandler
from onnx_jax.handlers.handler import onnx_op
from onnx_jax.pb_wrapper import OnnxNode


@onnx_op("BatchNormalization")
class BatchNormalization(BackendHandler):
    @classmethod
    def _common(cls, node, inputs, **kwargs):
        cls._rewrite(node)
        cls._prepare(node)

        return batchnorm

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
    def version_9(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def _rewrite(cls, node: OnnxNode):
        if 'epsilon' not in node.attrs:
            node.attrs['epsilon'] = 1e-5

    @classmethod
    def _prepare(cls, node: OnnxNode):
        args = list(inspect.signature(batchnorm).parameters.keys())
        attrs = [node.attrs.get(k, None) for k in args[node.len_inputs :]]
        node.attrs_list.extend(attrs)


@partial(jit, static_argnums=(5))
def batchnorm(x, s, bias, mean, var, epsilon=1e-5):
    dims_x = len(x.shape)
    dim_ones = (1,) * (dims_x - 2)
    s = s.reshape(-1, *dim_ones)
    bias = bias.reshape(-1, *dim_ones)
    mean = mean.reshape(-1, *dim_ones)
    var = var.reshape(-1, *dim_ones)
    ot = s * (x - mean) / lax.sqrt(var + epsilon) + bias
    return ot
