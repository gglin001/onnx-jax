import inspect
from functools import partial

import jax.numpy as jnp
from jax import jit
from jax.ops import index_update
from numpy.core.numeric import normalize_axis_index

from onnx_jax.handlers.backend_handler import BackendHandler
from onnx_jax.handlers.handler import onnx_op
from onnx_jax.pb_wrapper import OnnxNode


@onnx_op("Hardmax")
class Hardmax(BackendHandler):
    @classmethod
    def _common(cls, node: OnnxNode, **kwargs):
        cls._rewrite(node)
        cls._prepare(node)

        return onnx_hardmax

    @classmethod
    def version_1(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def version_11(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def version_13(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def _rewrite(cls, node: OnnxNode):
        if 'axis' not in node.attrs:
            node.attrs['axis'] = -1

    @classmethod
    def _prepare(cls, node: OnnxNode):
        args = list(inspect.signature(onnx_hardmax).parameters.keys())
        attrs = [node.attrs.get(k, None) for k in args[node.len_inputs :]]
        node.attrs_list.extend(attrs)


def _make_along_axis_idx(arr_shape, indices, axis):
    # copy from numpy
    # compute dimensions to iterate over
    # if not _nx.issubdtype(indices.dtype, _nx.integer):
    #     raise IndexError('`indices` must be an integer array')
    # if len(arr_shape) != indices.ndim:
    #     raise ValueError("`indices` and `arr` must have the same number of dimensions")
    shape_ones = (1,) * indices.ndim
    dest_dims = list(range(axis)) + [None] + list(range(axis + 1, indices.ndim))

    # build a fancy index, consisting of orthogonal aranges, with the
    # requested index inserted at the right location
    fancy_index = []
    for dim, n in zip(dest_dims, arr_shape):
        if dim is None:
            fancy_index.append(indices)
        else:
            ind_shape = shape_ones[:dim] + (-1,) + shape_ones[dim + 1 :]
            fancy_index.append(jnp.arange(n).reshape(ind_shape))
    return tuple(fancy_index)


@partial(jit, static_argnums=(2, 3))
def put_along_axis(arr, indices, values, axis):
    # copy from numpy
    # normalize inputs
    if axis is None:
        arr = arr.flat
        axis = 0
        arr_shape = (len(arr),)  # flatiter has no .shape
    else:
        axis = normalize_axis_index(axis, arr.ndim)
        arr_shape = arr.shape

    # use the fancy index
    return index_update(arr, _make_along_axis_idx(arr_shape, indices, axis), values)


@partial(jit, static_argnums=(1))
def onnx_hardmax(x, axis=-1):
    y = jnp.zeros_like(x)
    y = put_along_axis(y, jnp.expand_dims(jnp.argmax(x, axis), axis), 1.0, axis)
    return y
