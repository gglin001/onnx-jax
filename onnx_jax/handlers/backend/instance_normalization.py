import jax.numpy as jnp

from onnx_jax.handlers.backend_handler import BackendHandler
from onnx_jax.handlers.handler import onnx_op


@onnx_op("InstanceNormalization")
class InstanceNormalization(BackendHandler):

    @classmethod
    def _common(cls, node, inputs, **kwargs):
        return _instancenorm_test_mode(*inputs, **node.attrs)

    @classmethod
    def version_1(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def version_6(cls, node, **kwargs):
        return cls._common(node, **kwargs)


def _instancenorm_test_mode(x, scale, bias, epsilon=1e-5):
    ndim = x.ndim
    axis = tuple(range(2, ndim))
    mean = jnp.mean(x, axis=axis, keepdims=True)
    var = jnp.var(x, axis=axis, keepdims=True)
    dim_ones = (1,) * (ndim - 2)
    scale = scale.reshape(-1, *dim_ones)
    bias = bias.reshape(-1, *dim_ones)
    return [scale * (x - mean) / jnp.sqrt(var + epsilon) + bias]
