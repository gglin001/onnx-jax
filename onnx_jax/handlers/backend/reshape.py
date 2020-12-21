import jax.numpy as jnp
import numpy as onp

from onnx_jax.handlers.backend_handler import BackendHandler
from onnx_jax.handlers.handler import onnx_op


@onnx_op("Reshape")
class Reshape(BackendHandler):

    @classmethod
    def _common(cls, node, inputs, **kwargs):
        x = inputs[0]

        if cls.SINCE_VERSION == 1:
            shape = node.attrs["shape"]
        else:  # since_version >= 5
            shape = inputs[1]

        # ref https://github.com/onnx/onnx
        # replace zeros with corresponding dim size
        # we need to do this because np.reshape doesn't support 0
        zeros_index = onp.where(shape == 0)[0]
        if zeros_index.size:
            shape[zeros_index] = onp.array(x.shape)[zeros_index]
        return [jnp.reshape(x, shape)]

    @classmethod
    def version_1(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def version_5(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def version_13(cls, node, **kwargs):
        return cls._common(node, **kwargs)
