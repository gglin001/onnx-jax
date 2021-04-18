from jax import jit
from jax.nn import softplus

from onnx_jax.handlers.backend_handler import BackendHandler
from onnx_jax.handlers.handler import onnx_op
from onnx_jax.pb_wrapper import OnnxNode


@onnx_op("Softplus")
class Softplus(BackendHandler):
    @classmethod
    def _common(cls, node: OnnxNode, **kwargs):
        @jit
        def _softplus(x):
            return softplus(x)

        return _softplus

    @classmethod
    def version_1(cls, node, **kwargs):
        return cls._common(node, **kwargs)
