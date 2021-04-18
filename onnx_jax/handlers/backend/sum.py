import inspect
from functools import partial

import jax.numpy as jnp
from jax import jit

from onnx_jax.handlers.backend_handler import BackendHandler
from onnx_jax.handlers.handler import onnx_op
from onnx_jax.pb_wrapper import OnnxNode


@onnx_op("Sum")
class Sum(BackendHandler):
    @classmethod
    def _common(cls, node: OnnxNode, **kwargs):
        if node.len_inputs == 1:
            return onnx_sum_1
        elif node.len_inputs == 2:
            return onnx_sum_2
        elif node.len_inputs == 3:
            return onnx_sum_3
        elif node.len_inputs == 4:
            return onnx_sum_4
        elif node.len_inputs == 5:
            return onnx_sum_5
        else:
            raise NotImplemented('too many inputs')

    @classmethod
    def version_1(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def version_6(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def version_8(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def version_13(cls, node, **kwargs):
        return cls._common(node, **kwargs)


@jit
def onnx_sum_1(x):
    return x


@jit
def onnx_sum_2(x0, x1):
    return jnp.add(x0, x1)


@jit
def onnx_sum_3(x0, x1, x2):
    return jnp.add(jnp.add(x0, x1), x2)


@jit
def onnx_sum_4(x0, x1, x2, x3):
    return jnp.add(onnx_sum_3(x0, x1, x2), x3)


@jit
def onnx_sum_5(x0, x1, x2, x3, x4):
    return jnp.add(onnx_sum_4(x0, x1, x2, x3), x4)
