from functools import partial

import numpy as np

from onnx_jax.handlers.backend_handler import BackendHandler
from onnx_jax.handlers.handler import onnx_op


@onnx_op("Flatten")
class Flatten(BackendHandler):

    @classmethod
    def _common(cls, node, inputs, **kwargs):
        return onnx_flatten(*inputs, **node.attrs)

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
    def version_13(cls, node, **kwargs):
        return cls._common(node, **kwargs)


def onnx_flatten(x, axis=1):
    fun_prod = partial(np.prod, dtype=np.int32)
    new_shape = (fun_prod(x.shape[:axis]), fun_prod(x.shape[axis:]))
    return [x.reshape(new_shape)]
