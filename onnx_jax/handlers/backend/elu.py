import jax.numpy as jnp

from onnx_jax.handlers.backend_handler import BackendHandler
from onnx_jax.handlers.handler import onnx_op


@onnx_op("Elu")
class Elu(BackendHandler):

    @classmethod
    def _common(cls, node, inputs, **kwargs):
        return onnx_elu(*inputs, **node.attrs)

    @classmethod
    def version_1(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def version_6(cls, node, **kwargs):
        return cls._common(node, **kwargs)


def onnx_elu(x, alpha=1.0):
    return [jnp.clip(x, 0, jnp.inf) + (jnp.exp(jnp.clip(x, -jnp.inf, 0)) - 1) * alpha]
