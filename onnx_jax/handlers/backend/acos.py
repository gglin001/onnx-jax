from jax import jit, lax

from onnx_jax.handlers.backend_handler import BackendHandler
from onnx_jax.handlers.handler import onnx_op
from onnx_jax.pb_wrapper import OnnxNode


@onnx_op("Acos")
class Acos(BackendHandler):
    @classmethod
    def _common(cls, node: OnnxNode, **kwargs):
        @jit
        def _acos(x):
            return lax.acos(x)

        return _acos

    @classmethod
    def version_7(cls, node, **kwargs):
        return cls._common(node, **kwargs)
