import jax.numpy as jnp
from jax import jit

from onnx_jax.handlers.backend_handler import BackendHandler
from onnx_jax.handlers.handler import onnx_op
from onnx_jax.pb_wrapper import OnnxNode


@onnx_op("Mean")
class Mean(BackendHandler):
    @classmethod
    def _common(cls, node: OnnxNode, **kwargs):
        if node.len_inputs == 1:
            return onnx_mean_1
        elif node.len_inputs == 2:
            return onnx_mean_2
        elif node.len_inputs == 3:
            return onnx_mean_3
        elif node.len_inputs == 4:
            return onnx_mean_4
        else:
            raise NotImplemented("Max layer input size > 4")

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
def onnx_mean_1(x):
    return x


@jit
def onnx_mean_2(x0, x1):
    return jnp.add(x0, x1) / 2.0


@jit
def onnx_mean_3(x0, x1, x2):
    return jnp.add(jnp.add(x0, x1), x2) / 3.0


@jit
def onnx_mean_4(x0, x1, x2, x3):
    return jnp.add(jnp.add(jnp.add(x0, x1), x2), x3) / 4.0
