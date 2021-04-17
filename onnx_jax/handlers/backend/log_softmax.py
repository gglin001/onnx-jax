import inspect
from functools import partial

from jax import jit
from jax.nn import log_softmax

from onnx_jax.handlers.backend_handler import BackendHandler
from onnx_jax.handlers.handler import onnx_op
from onnx_jax.pb_wrapper import OnnxNode


@onnx_op("LogSoftmax")
class LogSoftmax(BackendHandler):
    @classmethod
    def _common(cls, node: OnnxNode, **kwargs):
        cls._rewrite(node)
        cls._prepare(node)

        return onnx_log_softmax

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
        args = list(inspect.signature(onnx_log_softmax).parameters.keys())
        attrs = [node.attrs.get(k, None) for k in args[node.len_inputs :]]
        node.attrs_list.extend(attrs)


@partial(jit, static_argnums=(1))
def onnx_log_softmax(x, axis=-1):
    return log_softmax(x, axis)
