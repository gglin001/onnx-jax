import jax.numpy as jnp
from jax import jit

from onnx_jax.handlers.backend_handler import BackendHandler
from onnx_jax.handlers.handler import onnx_op
from onnx_jax.pb_wrapper import OnnxNode


@onnx_op("GlobalMaxPool")
class GlobalMaxPool(BackendHandler):
    @classmethod
    def _common(cls, node: OnnxNode, **kwargs):
        return onnx_global_max_pool

    @classmethod
    def version_1(cls, node, **kwargs):
        return cls._common(node, **kwargs)


@jit
def onnx_global_max_pool(x):
    spatial_shape = jnp.ndim(x) - 2
    y = jnp.max(x, axis=tuple(range(spatial_shape, spatial_shape + 2)))
    for _ in range(spatial_shape):
        y = jnp.expand_dims(y, -1)
    return y
