import jax.numpy as jnp

from onnx_jax.handlers.backend_handler import BackendHandler
from onnx_jax.handlers.handler import onnx_op
from onnx_jax.pb_wrapper import OnnxNode


@onnx_op("NonZero")
class NonZero(BackendHandler):
    @classmethod
    def _common(cls, node: OnnxNode, **kwargs):
        # TODO non-jit
        def _nonzero(x):
            return jnp.asarray(jnp.nonzero(x))

        return _nonzero

    @classmethod
    def version_9(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def version_13(cls, node, **kwargs):
        return cls._common(node, **kwargs)
