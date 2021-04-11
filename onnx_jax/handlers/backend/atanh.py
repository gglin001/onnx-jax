from jax import jit, lax

from onnx_jax.handlers.backend_handler import BackendHandler
from onnx_jax.handlers.handler import onnx_op
from onnx_jax.pb_wrapper import OnnxNode


@onnx_op("Atanh")
class Atanh(BackendHandler):
    @classmethod
    def _common(cls, node: OnnxNode, **kwargs):
        @jit
        def _atanh(x):
            return lax.atanh(x)

        return _atanh

    @classmethod
    def version_9(cls, node, **kwargs):
        return cls._common(node, **kwargs)
