import inspect
from functools import partial

import jax.numpy as jnp
import numpy as np
from jax import jit, lax

from onnx_jax.handlers.backend_handler import BackendHandler
from onnx_jax.handlers.handler import onnx_op
from onnx_jax.pb_wrapper import OnnxNode


@onnx_op("AveragePool")
class AveragePool(BackendHandler):
    @classmethod
    def _common(cls, node: OnnxNode, **kwargs):
        cls._rewrite(node)
        cls._prepare(node)

        return onnx_avgpool

    @classmethod
    def version_1(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def version_7(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def version_10(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def version_11(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def _rewrite(cls, node: OnnxNode):
        if 'auto_pad' not in node.attrs:
            node.attrs['auto_pad'] = 'NOTSET'
        if 'ceil_mode' not in node.attrs:
            node.attrs['ceil_mode'] = 0
        if 'count_include_pad' not in node.attrs:
            node.attrs['count_include_pad'] = 0
        if 'pads' not in node.attrs:
            node.attrs['pads'] = None
        if 'strides' not in node.attrs:
            node.attrs['strides'] = None

    @classmethod
    def _prepare(cls, node: OnnxNode):
        args = list(inspect.signature(onnx_avgpool).parameters.keys())
        attrs = [node.attrs.get(k, None) for k in args[node.len_inputs :]]
        node.attrs_list.extend(attrs)


def pad_helper(input_rank, pads=None):
    pad_pairs = len(pads) // 2 if pads else 0

    pad_width = []
    for _ in range(input_rank - pad_pairs):
        pad_width.append((0, 0))
    for idx in range(pad_pairs):
        pad_width.append((pads[idx], pads[idx + pad_pairs]))
    return pad_width


@partial(jit, static_argnums=(1, 2, 3, 4, 5, 6))
def onnx_avgpool(
    x,
    kernel_shape,
    pads=None,
    strides=None,
    auto_pad='NOTSET',
    ceil_mode=0,
    count_include_pad=0,
):
    if ceil_mode != 0:
        raise NotImplemented('ceil_mode != 0')

    dims = (1,) * (x.ndim - len(kernel_shape)) + tuple(kernel_shape)
    strides = (
        ((1,) * (x.ndim - len(strides)) + tuple(strides)) if strides else (1,) * x.ndim
    )

    if auto_pad == "NOTSET":
        pads = pad_helper(x.ndim, pads) if pads else 'VALID'
    elif auto_pad == "SAME_UPPER":
        pads = "SAME"
    elif auto_pad == "VALID":
        pads = "VALID"
    elif auto_pad == "SAME_LOWER":
        raise NotImplemented("AveragePool with auto_pad `SAME_LOWER`")
    else:
        raise ValueError(f"Invalid auto_pad attribute: {auto_pad}")

    if count_include_pad == 0:
        one = jnp.ones_like(x, dtype=x.dtype)
        window_sizes = lax.reduce_window(one, 0.0, lax.add, dims, strides, pads)
    else:
        window_sizes = np.prod(kernel_shape)

    return [
        lax.reduce_window(x, 0.0, lax.add, dims, strides, pads, None, None)
        / window_sizes
    ]
