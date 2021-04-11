import jax.numpy as jnp
from onnx import mapping, numpy_helper

from onnx_jax.handlers.backend_handler import BackendHandler
from onnx_jax.handlers.handler import onnx_op
from onnx_jax.pb_wrapper import OnnxNode


@onnx_op("Constant")
class Constant(BackendHandler):
    @classmethod
    def _common(cls, node: OnnxNode, **kwargs):
        cls._prepare(node)

        def _constant(x):
            dtype = mapping.TENSOR_TYPE_TO_NP_TYPE[x.data_type]
            y = jnp.asarray(numpy_helper.to_array(x).astype(dtype)).reshape(x.dims)
            return y

        return _constant

    @classmethod
    def version_1(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def version_9(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def version_11(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def version_12(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def version_13(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def _prepare(cls, node: OnnxNode):
        attrs = [node.attrs.get('value')]
        node.attrs_list.extend(attrs)
