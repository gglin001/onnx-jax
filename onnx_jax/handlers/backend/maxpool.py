import inspect
from functools import partial

import jax.numpy as jnp
from jax import jit, lax

from onnx_jax.handlers.backend_handler import BackendHandler
from onnx_jax.handlers.handler import onnx_op
from onnx_jax.pb_wrapper import OnnxNode


@onnx_op("MaxPool")
class MaxPool(BackendHandler):
    @classmethod
    def _common(cls, node: OnnxNode, **kwargs):
        if len(node.outputs) > 1:
            raise Exception('MaxPool with indices is not supported')

        cls._rewrite(node)
        cls._prepare(node)

        return onnx_maxpool

    @classmethod
    def version_1(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def version_8(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def version_10(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def version_11(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def version_12(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def _rewrite(cls, node: OnnxNode):
        if 'auto_pad' not in node.attrs:
            node.attrs['auto_pad'] = 'NOTSET'
        if 'ceil_mode' not in node.attrs:
            node.attrs['ceil_mode'] = 0
        if 'storage_order' not in node.attrs:
            node.attrs['storage_order'] = 0

    @classmethod
    def _prepare(cls, node: OnnxNode):
        args = list(inspect.signature(onnx_maxpool).parameters.keys())
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


@partial(jit, static_argnums=(1, 2, 3, 4, 5, 6, 7))
def onnx_maxpool(
    x,
    kernel_shape,
    pads=None,
    strides=None,
    dilations=None,
    auto_pad='NOTSET',
    ceil_mode=0,
    storage_order=0,
):
    dims = (1,) * (x.ndim - len(kernel_shape)) + tuple(kernel_shape)
    strides = (
        ((1,) * (x.ndim - len(strides)) + tuple(strides)) if strides else (1,) * x.ndim
    )
    dilations = (
        ((1,) * (x.ndim - len(dilations)) + tuple(dilations))
        if dilations
        else (1,) * x.ndim
    )

    if auto_pad == "NOTSET":
        pads = pad_helper(x.ndim, pads) if pads else 'VALID'
    elif auto_pad == "SAME_UPPER":
        pads = "SAME"
    elif auto_pad == "VALID":
        pads = "VALID"
    elif auto_pad == "SAME_LOWER":
        raise NotImplemented("MaxPool with auto_pad `SAME_LOWER`")
    else:
        raise ValueError(f"Invalid auto_pad attribute: {auto_pad}")

    return lax.reduce_window(x, -jnp.inf, lax.max, dims, strides, pads, None, dilations)
