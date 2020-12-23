from onnx.mapping import TENSOR_TYPE_TO_NP_TYPE

from onnx_jax.handlers.backend_handler import BackendHandler
from onnx_jax.handlers.handler import onnx_op


@onnx_op("Cast")
class Cast(BackendHandler):

    @classmethod
    def _common(cls, node, inputs, **kwargs):
        return [inputs[0].astype(TENSOR_TYPE_TO_NP_TYPE[node.attrs['to']])]

    @classmethod
    def version_1(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def version_6(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def version_9(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def version_13(cls, node, **kwargs):
        return cls._common(node, **kwargs)
