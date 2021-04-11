from functools import partial

import jax.numpy as jnp
from jax import jit

from onnx_jax.handlers.backend_handler import BackendHandler
from onnx_jax.handlers.handler import onnx_op
from onnx_jax.pb_wrapper import OnnxNode


@onnx_op("Concat")
class Concat(BackendHandler):
    @classmethod
    def _common(cls, node: OnnxNode, **kwargs):
        cls._rewrite(node)
        cls._prepare(node)

        if node.len_inputs == 1:
            return onnx_concat_1
        elif node.len_inputs == 2:
            return onnx_concat_2
        elif node.len_inputs == 3:
            return onnx_concat_3
        elif node.len_inputs == 4:
            return onnx_concat_4
        elif node.len_inputs == 5:
            return onnx_concat_5
        else:
            raise NotImplemented('too many inputs')

    @classmethod
    def version_1(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def version_4(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def version_11(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def version_13(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def _rewrite(cls, node: OnnxNode):
        if 'axis' not in node.attrs:
            node.attrs['axis'] = 0

    @classmethod
    def _prepare(cls, node: OnnxNode):
        attrs = [node.attrs.get('axis')]
        node.attrs_list.extend(attrs)


@partial(jit, static_argnums=(1))
def onnx_concat_1(x, axis=0):
    return jnp.concatenate((x), axis)


@partial(jit, static_argnums=(2))
def onnx_concat_2(x0, x1, axis=0):
    return jnp.concatenate((x0, x1), axis)


@partial(jit, static_argnums=(3))
def onnx_concat_3(x0, x1, x2, axis=0):
    return jnp.concatenate((x0, x1, x2), axis)


@partial(jit, static_argnums=(4))
def onnx_concat_4(x0, x1, x2, x3, axis=0):
    return jnp.concatenate((x0, x1, x2, x3), axis)


@partial(jit, static_argnums=(5))
def onnx_concat_5(x0, x1, x2, x3, x4, axis=0):
    return jnp.concatenate((x0, x1, x2, x3, x4), axis)
