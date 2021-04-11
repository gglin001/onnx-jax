from jax import jit, lax

from onnx_jax.handlers.backend_handler import BackendHandler
from onnx_jax.handlers.handler import onnx_op
from onnx_jax.pb_wrapper import OnnxNode


@onnx_op("Acosh")
class Acos(BackendHandler):
    @classmethod
    def _common(cls, node: OnnxNode, **kwargs):
        @jit
        def _acosh(x):
            return lax.acosh(x)

        return _acosh

    @classmethod
    def version_9(cls, node, **kwargs):
        return cls._common(node, **kwargs)
