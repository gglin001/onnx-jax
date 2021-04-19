import inspect
from functools import partial

import jax.numpy as jnp
from jax import jit

from onnx_jax.handlers.backend_handler import BackendHandler
from onnx_jax.handlers.handler import onnx_op
from onnx_jax.pb_wrapper import OnnxNode

int32_max = jnp.iinfo(jnp.int32).max


@onnx_op("Slice")
class Slice(BackendHandler):
    @classmethod
    def _common(cls, node: OnnxNode, **kwargs):
        cls._rewrite(node)
        cls._prepare(node)

        def _slice(x, starts, ends, axes=None, steps=None):
            if axes is not None:
                axes = tuple(axes)
            if steps is not None:
                steps = tuple(steps)
            ends = [x if x < int32_max else int32_max for x in ends]
            return onnx_slice(x, tuple(starts), tuple(ends), axes, steps)

        return _slice

    @classmethod
    def version_1(cls, node, **kwargs):
        # TODO
        raise NotImplemented('Slice opset-v1')

    @classmethod
    def version_10(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def version_11(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def version_13(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def _rewrite(cls, node: OnnxNode):
        pass

    @classmethod
    def _prepare(cls, node: OnnxNode):
        args = list(inspect.signature(onnx_slice).parameters.keys())
        attrs = [node.attrs.get(k, None) for k in args[node.len_inputs :]]
        node.attrs_list.extend(attrs)


def axe_helper(ndim, x):
    return ndim + x if x < 0 else x


def start_helper(st, shape):
    return shape + st if st < 0 else min(st, shape)


def end_helper(end, shape):
    return shape + end if end < 0 else min(end, shape)


@partial(jit, static_argnums=(1, 2, 3, 4))
def onnx_slice(x, starts, ends, axes=None, steps=None):
    ndim = x.ndim
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

    slices = [
        slice(_st, _end, _step)
        for _st, _end, _step in zip(starts_new, ends_new, steps_new)
    ]
    return x[tuple(slices)]
