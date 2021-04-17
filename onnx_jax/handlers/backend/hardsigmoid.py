import inspect

import jax.numpy as jnp
from jax import jit

from onnx_jax.handlers.backend_handler import BackendHandler
from onnx_jax.handlers.handler import onnx_op
from onnx_jax.pb_wrapper import OnnxNode


@onnx_op("HardSigmoid")
class HardSigmoid(BackendHandler):
    @classmethod
    def _common(cls, node: OnnxNode, **kwargs):
        cls._rewrite(node)
        cls._prepare(node)

        return onnx_hardsigmoid

    @classmethod
    def version_1(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def version_6(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def _rewrite(cls, node: OnnxNode):
        if 'alpha' not in node.attrs:
            node.attrs['alpha'] = 0.2
        if 'beta' not in node.attrs:
            node.attrs['beta'] = 0.5

    @classmethod
    def _prepare(cls, node: OnnxNode):
        args = list(inspect.signature(onnx_hardsigmoid).parameters.keys())
        attrs = [node.attrs.get(k, None) for k in args[node.len_inputs :]]
        node.attrs_list.extend(attrs)


@jit
def onnx_hardsigmoid(x, alpha=0.2, beta=0.5):
    return jnp.clip(x * alpha + beta, 0, 1)
