import inspect
from functools import partial

import jax.numpy as jnp
from jax import jit, lax
from onnx_jax.handlers.backend_handler import BackendHandler
from onnx_jax.handlers.handler import onnx_op
from onnx_jax.pb_wrapper import OnnxNode


@onnx_op("GlobalAveragePool")
class GlobalAveragePool(BackendHandler):
    @classmethod
    def _common(cls, node, inputs, **kwargs):
        return global_average_pool

    @classmethod
    def version_1(cls, node, **kwargs):
        return cls._common(node, **kwargs)


@jit
def global_average_pool(x):
    spatial_dim = jnp.ndim(x) - 2
    y = jnp.mean(x, axis=tuple(range(spatial_dim, spatial_dim + 2)))
    for _ in range(spatial_dim):
        y = jnp.expand_dims(y, -1)
    return y
