from jax import lax
import jax.numpy as jnp

from onnx_jax.handlers.backend_handler import BackendHandler
from onnx_jax.handlers.handler import onnx_op


@onnx_op("Conv")
class Conv(BackendHandler):
    @classmethod
    def _common(cls, node, inputs, **kwargs):
        return onnx_conv(*inputs, **node.attrs)

    @classmethod
    def version_1(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def version_11(cls, node, **kwargs):
        return cls._common(node, **kwargs)


def pad_helper(data, pads, mode, constant_values=0.0):
    input_rank = data.ndim
    pad_pairs = len(pads) // 2

    pad_width = []
    for _ in range(input_rank - pad_pairs):
        pad_width.append((0, 0))
    for idx in range(pad_pairs):
        pad_width.append((pads[idx], pads[idx + pad_pairs]))

    if mode == 'constant':
        return jnp.pad(
            data, pad_width=pad_width, mode=mode, constant_values=constant_values
        )

    return jnp.pad(data, pad_width=pad_width, mode=mode)


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
        if pads is not None and pads != [0, 0] * spatial_size:
            x = pad_helper(x, pads, 'constant', 0.0)
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

    return [
        lax.conv_general_dilated(
            x, w, strides, pad_mode, lhs_dilation, rhs_dilation, None, group, 1
        )
        + b
    ]
