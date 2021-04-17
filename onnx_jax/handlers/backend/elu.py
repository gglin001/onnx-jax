import inspect

import jax.numpy as jnp
from jax import jit

from onnx_jax.handlers.backend_handler import BackendHandler
from onnx_jax.handlers.handler import onnx_op
from onnx_jax.pb_wrapper import OnnxNode


@onnx_op("Elu")
class Elu(BackendHandler):
    @classmethod
    def _common(cls, node: OnnxNode, **kwargs):
        cls._rewrite(node)
        cls._prepare(node)

        return onnx_elu

    @classmethod
    def version_1(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def version_6(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def _rewrite(cls, node: OnnxNode):
        if 'alpha' not in node.attrs:
            node.attrs['alpha'] = 1.0

    @classmethod
    def _prepare(cls, node: OnnxNode):
        args = list(inspect.signature(onnx_elu).parameters.keys())
        attrs = [node.attrs.get(k, None) for k in args[node.len_inputs :]]
        node.attrs_list.extend(attrs)


@jit
def onnx_elu(x, alpha=1.0):
    return jnp.clip(x, 0, jnp.inf) + (jnp.exp(jnp.clip(x, -jnp.inf, 0)) - 1) * alpha
