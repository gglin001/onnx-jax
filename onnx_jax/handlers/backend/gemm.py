import inspect
from functools import partial

import jax.numpy as jnp
from jax import jit

from onnx_jax.handlers.backend_handler import BackendHandler
from onnx_jax.handlers.handler import onnx_op
from onnx_jax.pb_wrapper import OnnxNode


@onnx_op("Gemm")
class Gemm(BackendHandler):
    @classmethod
    def _common(cls, node: OnnxNode, **kwargs):
        cls._rewrite(node)
        cls._prepare(node)

        return onnx_gemm

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
    def version_11(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def version_13(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def _rewrite(cls, node: OnnxNode):
        if 'alpha' not in node.attrs:
            node.attrs['alpha'] = 1.0
        if 'beta' not in node.attrs:
            node.attrs['beta'] = 1.0
        if 'transA' not in node.attrs:
            node.attrs['transA'] = 0
        if 'transB' not in node.attrs:
            node.attrs['transB'] = 0

    @classmethod
    def _prepare(cls, node: OnnxNode):
        args = list(inspect.signature(onnx_gemm).parameters.keys())
        attrs = [node.attrs.get(k, None) for k in args[node.len_inputs :]]
        node.attrs_list.extend(attrs)


@partial(jit, static_argnums=(3, 4, 5, 6))
def onnx_gemm(a, b, c=None, alpha=1.0, beta=1.0, transA=0, transB=0):
    a = a if transA == 0 else a.T
    b = b if transB == 0 else b.T
    if c is None:
        return alpha * jnp.dot(a, b) + beta
    else:
        return alpha * jnp.dot(a, b) + beta * c
