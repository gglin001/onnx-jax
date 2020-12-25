from jax import lax
import jax.numpy as jnp

from onnx_jax.handlers.backend_handler import BackendHandler
from onnx_jax.handlers.handler import onnx_op


@onnx_op("ConvTranspose")
class ConvTranspose(BackendHandler):

    @classmethod
    def _common(cls, node, inputs, **kwargs):
        return onnx_conv_transpose(*inputs, **node.attrs)

    @classmethod
    def version_1(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def version_11(cls, node, **kwargs):
        return cls._common(node, **kwargs)


def pad_helper(data, pads, mode='constant', constant_values=0.0):
    input_rank = data.ndim
    pad_pairs = len(pads) // 2

    pad_width = []
    for _ in range(input_rank - pad_pairs):
        pad_width.append((0, 0))
    for idx in range(pad_pairs):
        pad_width.append((pads[idx], pads[idx + pad_pairs]))

    if mode == 'constant':
        return jnp.pad(data, pad_width=pad_width, mode=mode, constant_values=constant_values)

    return jnp.pad(data, pad_width=pad_width, mode=mode)


def onnx_conv_transpose(x, w, b=None, auto_pad='NOTSET', dilations=None, group=1, kernel_shape=None,
                        output_padding=None, output_shape=None, pads=None, strides=None, **kwargs):

    kernel_shape = kernel_shape or w.shape
    spatial_size = w.ndim - 2
    strides = strides or [1] * spatial_size
    rhs_dilation = dilations or [1] * (w.ndim - 2)

    # pad
    if auto_pad == "NOTSET":
        if pads is None:
            pad_mode = 'VALID'
        elif pads == 'VALID':
            pad_mode = 'VALID'
        elif pads == [0, 0] * spatial_size:
            pad_mode = pads
        else:
            pad_mode = []
            pad_pairs = len(pads) // 2
            for idx in range(pad_pairs):
                pad_mode.append((pads[idx], pads[idx + pad_pairs]))
    elif auto_pad == "SAME_UPPER":
        pad_mode = "SAME"
    elif auto_pad == "VALID":
        pad_mode = "VALID"
    elif auto_pad == "SAME_LOWER":
        raise NotImplemented("Conv with auto_pad `SAME_LOWER`")
    else:
        raise ValueError("Invalid auto_pad attribute: {}".format(auto_pad))

    if b is not None:
        b = b.reshape([1, w.shape[0]] + [1] * spatial_size)
    else:
        b = 0

    res = lax.conv_transpose(lhs=x,
                             rhs=w,
                             strides=strides,
                             padding=pad_mode,
                             rhs_dilation=rhs_dilation,
                             dimension_numbers=('NCHW', 'OIHW', 'NCHW'),
                             transpose_kernel=True,
                             precision=None)

    # change output_padding order
    # TODO
    output_padding = [0, 0, 0, 0] if output_padding is None else [0, 0, output_padding[0], output_padding[1]]
    if output_shape is not None:
        need_append_output_pad = True
        for spatial_idx in range(spatial_size):
            total_pad = output_padding[spatial_idx] + output_padding[spatial_size + spatial_idx]
            shape_diff = output_shape[spatial_idx] - res.shape[spatial_idx + 2] - total_pad
            if shape_diff == 0:
                need_append_output_pad = False
            else:
                need_append_output_pad = True

        if need_append_output_pad:
            for spatial_idx in range(spatial_size):
                shape_diff = output_shape[spatial_idx] - res.shape[spatial_idx + 2]
                if shape_diff < 0:
                    raise Exception('output_sahpe can not samller than lax.conv_transpose output shape')
                else:
                    output_padding[spatial_idx + spatial_size] += shape_diff

    if output_padding != [0, 0, 0, 0]:
        res = pad_helper(res, output_padding)

    return [res + b]
