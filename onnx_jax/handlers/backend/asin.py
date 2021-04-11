from jax import jit, lax
from onnx_jax.handlers.backend_handler import BackendHandler
from onnx_jax.handlers.handler import onnx_op
from onnx_jax.pb_wrapper import OnnxNode


@onnx_op("Asin")
class Asin(BackendHandler):
    @classmethod
    def _common(cls, node: OnnxNode, **kwargs):
        @jit
        def _asin(x):
            return lax.asin(x)

        return _asin

    @classmethod
    def version_7(cls, node, **kwargs):
        return cls._common(node, **kwargs)
