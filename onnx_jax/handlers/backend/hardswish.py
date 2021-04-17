import inspect

import jax.numpy as jnp
from jax import jit

from onnx_jax.handlers.backend_handler import BackendHandler
from onnx_jax.handlers.handler import onnx_op
from onnx_jax.pb_wrapper import OnnxNode


@onnx_op("HardSwish")
class HardSwish(BackendHandler):
    @classmethod
    def _common(cls, node: OnnxNode, **kwargs):
        cls._rewrite(node)
        cls._prepare(node)

        return onnx_hardswish

    @classmethod
    def version_1(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def version_14(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def _rewrite(cls, node: OnnxNode):
        if 'alpha' not in node.attrs:
            node.attrs['alpha'] = 1.0 / 6.0
        if 'beta' not in node.attrs:
            node.attrs['beta'] = 0.5

    @classmethod
    def _prepare(cls, node: OnnxNode):
        args = list(inspect.signature(onnx_hardswish).parameters.keys())
        attrs = [node.attrs.get(k, None) for k in args[node.len_inputs :]]
        node.attrs_list.extend(attrs)


@jit
def onnx_hardswish(x, alpha=0.1666667, beta=0.5):
    return x * jnp.clip(x * alpha + beta, 0, 1)
