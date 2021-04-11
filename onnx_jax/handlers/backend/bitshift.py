import inspect
from functools import partial

from onnx_jax.handlers.backend_handler import BackendHandler
from onnx_jax.handlers.handler import onnx_op
from onnx_jax.pb_wrapper import OnnxNode


@onnx_op("BitShift")
class BitShift(BackendHandler):
    @classmethod
    def _common(cls, node: OnnxNode, **kwargs):
        cls._prepare(node)

        return onnx_bitshift

    @classmethod
    def version_11(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def _prepare(cls, node: OnnxNode):
        args = list(inspect.signature(onnx_bitshift).parameters.keys())
        attrs = [node.attrs.get(k, None) for k in args[node.len_inputs :]]
        node.attrs_list.extend(attrs)


@partial(jit, static_argnums=(2))
def onnx_bitshift(x, y, direction):
    if direction == "LEFT":
        z = x << y
    elif direction == "RIGHT":
        z = x >> y
    else:
        raise ValueError(f"direction is unknown: {direction}")
    return z
