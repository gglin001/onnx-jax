import inspect
from functools import partial

from jax import jit, lax
from onnx_jax.handlers.backend_handler import BackendHandler
from onnx_jax.handlers.handler import onnx_op
from onnx_jax.pb_wrapper import OnnxNode


@onnx_op("Conv")
class Conv(BackendHandler):
    @classmethod
    def _common(cls, node, **kwargs):
        cls._rewrite(node)
        cls._prepare(node)

        return onnx_conv

    @classmethod
    def version_1(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def version_11(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def _rewrite(cls, node: OnnxNode):
        if 'group' not in node.attrs:
            node.attrs['group'] = 1
        if 'pads' in node.attrs:
            pads = node.attrs['pads']
            pads_new = [
                (0, 0, 0),
                (0, 0, 0),
                (pads[0], pads[2], 0),
                (pads[1], pads[3], 0),
            ]
            node.attrs['pads'] = tuple(pads_new)

    @classmethod
    def _prepare(cls, node: OnnxNode):
        args = list(inspect.signature(onnx_conv).parameters.keys())
        attrs = [node.attrs.get(k, None) for k in args[node.len_inputs :]]
        node.attrs_list.extend(attrs)


@partial(jit, static_argnums=(3, 4, 5, 6, 7, 8))
def onnx_conv(
    x,
    w,
    b=None,
    group=1,
    kernel_shape=None,
    pads=None,
    strides=None,
    dilations=None,
    auto_pad=None,
):
    kernel_shape = kernel_shape or w.shape
    spatial_size = w.ndim - 2
    strides = strides or [1] * spatial_size

    # TODO some pad does not need a PadOp
    if not auto_pad or auto_pad == "NOTSET":
        if pads is not None:
            x = lax.pad(x, 0.0, pads)
        pad_mode = "VALID"
    elif auto_pad == "SAME_UPPER":
        pad_mode = "SAME"
    elif auto_pad == "VALID":
        pad_mode = "VALID"
    elif auto_pad == "SAME_LOWER":
        raise NotImplemented("Conv with auto_pad `SAME_LOWER`")
    else:
        raise ValueError("Invalid auto_pad attribute: {}".format(auto_pad))

    lhs_dilation = [1] * (w.ndim - 2)
    rhs_dilation = dilations or [1] * (w.ndim - 2)

    if b is not None:
        b = b.reshape([1, w.shape[0]] + [1] * spatial_size)
    else:
        b = 0

    out = lax.conv_general_dilated(
        x, w, strides, pad_mode, lhs_dilation, rhs_dilation, None, group, 1
    )
    return out + b
