from onnx_jax.handlers.backend_handler import BackendHandler
from onnx_jax.handlers.handler import onnx_op


@onnx_op("Slice")
class Slice(BackendHandler):

    @classmethod
    def _common(cls, node, inputs, **kwargs):
        return onnx_slice_v10(*inputs, **node.attrs)

    @classmethod
    def version_1(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def version_10(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def version_11(cls, node, **kwargs):
        return cls.version_10(node, **kwargs)

    @classmethod
    def version_13(cls, node, **kwargs):
        return cls.version_10(node, **kwargs)


def axe_helper(ndim, x): return ndim + x if x < 0 else x


def start_helper(st, shape): return shape + st if st < 0 else min(st, shape)


def end_helper(end, shape): return shape + end if end < 0 else min(end, shape)


def onnx_slice_v10(data, starts, ends, axes=None, steps=None, **kwargs):
    ndim = data.ndim

    starts_new, ends_new, axes_new, steps_new = [], [], [], []
    if axes is None and steps is None:
        steps_new = [None] * ndim
        starts_new, ends_new = starts, ends
    elif axes is not None and steps is None:
        steps_new = [None] * ndim
        for dim in range(ndim):
            for st, end, axe in zip(starts, ends, axes):
                axe = axe_helper(ndim, axe)
                if axe == dim:
                    starts_new.append(st)
                    ends_new.append(end)
                    axes_new.append(axe)
                    break
            if not axes_new or axes_new[-1] != dim:
                starts_new.append(None)
                ends_new.append(None)
                axes_new.append(dim)
    else:
        for dim in range(ndim):
            for st, end, axe, step in zip(starts, ends, axes, steps):
                axe = axe_helper(ndim, axe)
                if axe == dim:
                    starts_new.append(st)
                    ends_new.append(end)
                    axes_new.append(axe)
                    steps_new.append(step)
                    break
            if not axes_new or axes_new[-1] != dim:
                starts_new.append(None)
                ends_new.append(None)
                axes_new.append(dim)
                steps_new.append(None)

    slices = [slice(_st, _end, _step) for _st, _end, _step in zip(starts_new, ends_new, steps_new)]
    return [data.__getitem__(tuple(slices))]
