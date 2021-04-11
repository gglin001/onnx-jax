from functools import partial
import inspect
from jax import jit
import jax.numpy as jnp

from onnx_jax.handlers.backend_handler import BackendHandler
from onnx_jax.handlers.handler import onnx_op
from onnx_jax.pb_wrapper import OnnxNode


@onnx_op("Add")
class Add(BackendHandler):
    @classmethod
    def _common(cls, node: OnnxNode, **kwargs):
        cls._rewrite(node)
        cls._prepare(node)

        return onnx_add

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
    def version_13(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def _rewrite(cls, node: OnnxNode):
        if 'broadcast' not in node.attrs:
            node.attrs['broadcast'] = False

    @classmethod
    def _prepare(cls, node: OnnxNode):
        args = list(inspect.signature(onnx_add).parameters.keys())
        attrs = [node.attrs.get(k, None) for k in args[node.len_inputs :]]
        node.attrs_list.extend(attrs)


@partial(jit, static_argnums=(2, 3))
def onnx_add(a, b, axis=None, broadcast=False):
    if broadcast:
        b_shape = []
        b_shape.extend(a.shape[:axis])
        b_shape.append(a.shape[axis])
        b_shape.extend([1] * len(a.shape[axis + 1 :]))
        b = jnp.reshape(b, b_shape)
    elif len(a.shape) != len(b.shape):
        b_shape = [1] * len(a.shape)
        b_shape[1] = -1
        b = jnp.reshape(b, b_shape)

    return jnp.add(a, b)
