from jax import jit
from jax.nn import soft_sign

from onnx_jax.handlers.backend_handler import BackendHandler
from onnx_jax.handlers.handler import onnx_op
from onnx_jax.pb_wrapper import OnnxNode


@onnx_op("Softsign")
class Softsign(BackendHandler):
    @classmethod
    def _common(cls, node: OnnxNode, **kwargs):
        @jit
        def _soft_sign(x):
            return soft_sign(x)

        return _soft_sign

    @classmethod
    def version_1(cls, node, **kwargs):
        return cls._common(node, **kwargs)
