import numpy as np
from onnx import TensorProto

from onnx_jax.handlers.backend_handler import BackendHandler
from onnx_jax.handlers.handler import onnx_op


TENSOR_TYPE_TO_JNP_TYPE = {
    int(TensorProto.FLOAT): np.dtype('float32'),
    int(TensorProto.UINT8): np.dtype('uint8'),
    int(TensorProto.INT8): np.dtype('int8'),
    int(TensorProto.UINT16): np.dtype('uint16'),
    int(TensorProto.INT16): np.dtype('int16'),
    int(TensorProto.INT32): np.dtype('int32'),
    # for jax, int64 --> int32
    int(TensorProto.INT64): np.dtype('int32'),
    int(TensorProto.BOOL): np.dtype('bool'),
    int(TensorProto.FLOAT16): np.dtype('float16'),
    int(TensorProto.DOUBLE): np.dtype('float64'),
    int(TensorProto.COMPLEX64): np.dtype('complex64'),
    int(TensorProto.COMPLEX128): np.dtype('complex128'),
    int(TensorProto.UINT32): np.dtype('uint32'),
    int(TensorProto.UINT64): np.dtype('uint64'),
    int(TensorProto.STRING): np.dtype(np.object),
}


@onnx_op("Cast")
class Cast(BackendHandler):
    @classmethod
    def _common(cls, node, inputs, **kwargs):
        return [inputs[0].astype(TENSOR_TYPE_TO_JNP_TYPE[node.attrs['to']])]

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
