import inspect
from functools import partial

import jax.numpy as jnp
from jax import jit

from onnx_jax.handlers.backend_handler import BackendHandler
from onnx_jax.handlers.handler import onnx_op
from onnx_jax.pb_wrapper import OnnxNode


@onnx_op("Squeeze")
class Squeeze(BackendHandler):
    @classmethod
    def _common(cls, node: OnnxNode, **kwargs):
        cls._rewrite(node)
        cls._prepare(node)

        def _squeeze(x, axes):
            return onnx_squeeze(x, tuple(axes))

        return _squeeze

    @classmethod
    def version_1(cls, node, **kwargs):
        # axes is attr
        return cls._common(node, **kwargs)

    @classmethod
    def version_11(cls, node, **kwargs):
        # axes is attr
        return cls._common(node, **kwargs)

    @classmethod
    def version_13(cls, node, **kwargs):
        # axes is input
        return cls._common(node, **kwargs)

    @classmethod
    def _rewrite(cls, node: OnnxNode):
        if 'axes' in node.attrs:
            axes = node.attrs['axes']
            node.attrs['axes'] = tuple(axes)

    @classmethod
    def _prepare(cls, node: OnnxNode):
        args = list(inspect.signature(onnx_squeeze).parameters.keys())
        attrs = [node.attrs.get(k, None) for k in args[node.len_inputs :]]
        node.attrs_list.extend(attrs)


@partial(jit, static_argnums=(1))
def onnx_squeeze(x, axes=None):
    return jnp.squeeze(x, axes)
