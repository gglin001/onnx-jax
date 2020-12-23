import jax.numpy as jnp

from onnx_jax.handlers.backend_handler import BackendHandler
from onnx_jax.handlers.handler import onnx_op


@onnx_op("Pad")
class Pad(BackendHandler):

    @classmethod
    def _common(cls, node, inputs, **kwargs):
        return pad_impl(*inputs, **node.attrs)

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


def pad_impl(data, pads, constant_value=0.0, mode='constant', **kwargs):
    input_rank = data.ndim
    if input_rank * 2 != pads.size:
        raise Exception('The number of elements in raw_pads should be 2 * data_rank')

    # re-order to np.pad accepted order ((x1_begin, x1_end), (x2_begin, x2_end), ...)
    pad_width = ()
    for i in range(int(pads.size / 2)):
        pad_width += ((pads[i], pads[i + input_rank])),

    if mode == 'constant':
        return [jnp.pad(data, pad_width, mode, constant_values=constant_value)]
    return [jnp.pad(data, pad_width, mode)]
