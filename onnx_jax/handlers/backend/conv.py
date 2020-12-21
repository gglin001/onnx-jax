from jax import lax

from onnx_jax.handlers.backend_handler import BackendHandler
from onnx_jax.handlers.handler import onnx_op


@onnx_op("Conv")
class Conv(BackendHandler):

    @classmethod
    def _common(cls, node, inputs, transpose=False, **kwargs):
        return onnx_conv(*inputs, **node.attrs)

    @classmethod
    def version_1(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def version_11(cls, node, **kwargs):
        return cls._common(node, **kwargs)


# TODO transpose depthwise group
def onnx_conv(x, w, b=0, group=1, kernel_shape=None, pads=None, strides=None,
              dilations=None, auto_pad=None):
    """Numpy-backed implementation of ONNX Conv op."""
    assert group == 1
    kernel_shape = kernel_shape or w.shape
    strides = strides or [1] * (w.ndim - 2)
    if auto_pad:
        auto_pad = 'SAME' if auto_pad.startswith('SAME') else 'VALID'
        pads = lax.padtype_to_pads(x.shape[2:], w.shape[2:], strides, auto_pad)
    else:
        pads = pads or [0] * (w.ndim - 2)
        if len(pads) == 4:
            pads = [(pads[0], pads[1]), (pads[2], pads[3])]
    lhs_dilation = [1] * (w.ndim - 2)
    rhs_dilation = dilations or [1] * (w.ndim - 2)
    return [lax.conv_with_general_padding(x, w, strides, pads, lhs_dilation, rhs_dilation) + b]
