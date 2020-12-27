from jax import lax

from onnx_jax.handlers.backend_handler import BackendHandler
from onnx_jax.handlers.handler import onnx_op


@onnx_op("Split")
class Split(BackendHandler):

    @classmethod
    def _common(cls, node, inputs, **kwargs):
        return onnx_split(node, *inputs, **node.attrs)

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


def onnx_split(node, x, split=None, axis=0):
    n_out = len(node.outputs)
    split = [x.shape[axis] // n_out] * n_out if split is None else split

    starts = []
    ends = []
    starts.append([0] * x.ndim)
    for idx in range(1, n_out):
        st = [0] * x.ndim
        st[axis] = sum(split[:idx])
        starts.append(st)

        en = list(x.shape)
        en[axis] = sum(split[:idx])
        ends.append(en)
    ends.append(list(x.shape))

    return [lax.slice(x, start, end) for start, end in zip(starts, ends)]
