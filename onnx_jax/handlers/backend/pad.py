import inspect
from functools import partial

import jax.numpy as jnp
from jax import jit

from onnx_jax.handlers.backend_handler import BackendHandler
from onnx_jax.handlers.handler import onnx_op
from onnx_jax.pb_wrapper import OnnxNode


@onnx_op("Pad")
class Pad(BackendHandler):
    @classmethod
    def _common(cls, node: OnnxNode, **kwargs):
        cls._rewrite(node)
        cls._prepare(node)

        def _pad(x, pads, constant_value, mode):
            return onnx_pad(x, tuple(pads), float(constant_value), mode)

        return _pad

    @classmethod
    def version_1(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def version_2(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def version_11(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def version_13(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def _rewrite(cls, node: OnnxNode):
        if 'mode' not in node.attrs:
            node.attrs['mode'] = 'constant'
        if 'constant_value' not in node.attrs:
            node.attrs['constant_value'] = 0.0

        # opset-v1
        if 'paddings' in node.attrs:
            node.attrs['pads'] = node.attrs['paddings']

        # opset-v1 & v2
        if 'value' in node.attrs:
            node.attrs['constant_value'] = node.attrs['value']

    @classmethod
    def _prepare(cls, node: OnnxNode):
        args = list(inspect.signature(onnx_pad).parameters.keys())
        attrs = [node.attrs.get(k, None) for k in args[node.len_inputs :]]
        node.attrs_list.extend(attrs)


@partial(jit, static_argnums=(1, 2, 3))
def onnx_pad(x, pads, constant_value=0.0, mode='constant'):
    input_rank = x.ndim
    if input_rank * 2 != jnp.size(pads):
        raise Exception('The number of elements in raw_pads should be 2 * data_rank')

    # re-order to np.pad accepted order ((x1_begin, x1_end), (x2_begin, x2_end), ...)
    pad_width = ()
    for i in range(int(jnp.size(pads) / 2)):
        pad_width += (((pads[i], pads[i + input_rank])),)

    if mode == 'constant':
        return jnp.pad(x, pad_width, mode, constant_values=constant_value)
    return jnp.pad(x, pad_width, mode)
