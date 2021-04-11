import jax.numpy as jnp
from onnx import mapping, numpy_helper

from onnx_jax.handlers.backend_handler import BackendHandler
from onnx_jax.handlers.handler import onnx_op


@onnx_op("Constant")
class Constant(BackendHandler):
    @classmethod
    def _common(cls, node, **kwargs):
        attr_value = node.attrs["value"]
        dtype = mapping.TENSOR_TYPE_TO_NP_TYPE[attr_value.data_type]
        return [jnp.asarray(numpy_helper.to_array(attr_value).astype(dtype))]

    @classmethod
    def version_1(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def version_9(cls, node, **kwargs):
        return cls._common(node, **kwargs)
