import inspect
from functools import partial

from jax import jit, lax

from onnx_jax.handlers.backend_handler import BackendHandler
from onnx_jax.handlers.handler import onnx_op
from onnx_jax.pb_wrapper import OnnxNode


@onnx_op("Split")
class Split(BackendHandler):
    @classmethod
    def _common(cls, node: OnnxNode, **kwargs):
        cls._rewrite(node)
        cls._prepare(node)

        def _split(x, split, axis, n_out):
            if split is None:
                return onnx_split(x, split, axis, n_out)
            else:
                return onnx_split(x, tuple(split), axis, n_out)

        return _split

    @classmethod
    def version_1(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def version_2(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def version_11(cls, node, **kwargs):
        # split is attr
        return cls._common(node, **kwargs)

    @classmethod
    def version_13(cls, node, **kwargs):
        # split is input
        return cls._common(node, **kwargs)

    @classmethod
    def _rewrite(cls, node: OnnxNode):
        if 'axis' not in node.attrs:
            node.attrs['axis'] = 0
        if 'split' in node.attrs:
            split = node.attrs['split']
            node.attrs['split'] = tuple(split)
        node.attrs['n_out'] = node.len_outputs

    @classmethod
    def _prepare(cls, node: OnnxNode):
        args = list(inspect.signature(onnx_split).parameters.keys())
        attrs = [node.attrs.get(k, None) for k in args[node.len_inputs :]]
        node.attrs_list.extend(attrs)


@partial(jit, static_argnums=(1, 2, 3))
def onnx_split(x, split=None, axis=0, n_out=None):
    if split is None:
        split = [x.shape[axis] // n_out] * n_out

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
