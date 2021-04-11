import inspect

import jax.numpy as jnp
from jax import jit

from onnx_jax.handlers.backend_handler import BackendHandler
from onnx_jax.handlers.handler import onnx_op
from onnx_jax.pb_wrapper import OnnxNode


@onnx_op("Celu")
class Celu(BackendHandler):
    @classmethod
    def _common(cls, node: OnnxNode, **kwargs):
        cls._prepare(node)

        return onnx_celu

    @classmethod
    def version_12(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def _prepare(cls, node: OnnxNode):
        args = list(inspect.signature(onnx_celu).parameters.keys())
        attrs = [node.attrs.get(k, None) for k in args[node.len_inputs :]]
        node.attrs_list.extend(attrs)


@jit
def onnx_celu(x, alpha):
    p = jnp.maximum(0, x)
    n = jnp.minimum(0, alpha * (jnp.exp(x / alpha) - 1))
    y = p + n
    return y
