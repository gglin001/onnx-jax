from jax import lax

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
    data_shape = data.shape

    if axes is None and steps is None:
        starts_new, ends_new = [], []
        for dim in range(ndim):
            starts_new.append(start_helper(starts[dim], data_shape[dim]))
            ends_new.append(end_helper(ends[dim], data_shape[dim]))
        return [lax.slice(data, starts_new, ends_new)]
    elif axes is not None and steps is None:
        starts_new, ends_new, axes_new = [], [], []
        for dim in range(ndim):
            for st, end, axe in zip(starts, ends, axes):
                axe = axe_helper(ndim, axe)
                if axe == dim:
                    starts_new.append(start_helper(st, data_shape[dim]))
                    ends_new.append(end_helper(end, data_shape[dim]))
                    axes_new.append(axe)
                    break
            if not axes_new or axes_new[-1] != dim:
                starts_new.append(0)
                ends_new.append(data_shape[dim])
                axes_new.append(dim)
        return [lax.slice(data, starts_new, ends_new)]
    else:
        for step in steps:
            if step < 0:
                raise NotImplemented('Resize with negative value')

        starts_new, ends_new, axes_new, steps_new = [], [], [], []
        for dim in range(ndim):
            for st, end, axe, step in zip(starts, ends, axes, steps):
                axe = axe_helper(ndim, axe)
                if axe == dim:
                    starts_new.append(start_helper(st, data_shape[dim]))
                    ends_new.append(end_helper(end, data_shape[dim]))
                    axes_new.append(axe)
                    steps_new.append(step)
                    break
            if not axes_new or axes_new[-1] != dim:
                starts_new.append(0)
                ends_new.append(data_shape[dim])
                axes_new.append(dim)
                steps_new.append(1)
        return [lax.slice(data, starts_new, ends_new, steps_new)]
