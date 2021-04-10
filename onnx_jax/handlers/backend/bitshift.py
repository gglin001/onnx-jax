from onnx_jax.handlers.backend_handler import BackendHandler
from onnx_jax.handlers.handler import onnx_op


@onnx_op("BitShift")
class BitShift(BackendHandler):
    @classmethod
    def _common(cls, node, inputs, **kwargs):
        return (
            [inputs[0] << inputs[1]]
            if node.attrs['direction'] == "LEFT"
            else [inputs[0] >> inputs[1]]
        )

    @classmethod
    def version_11(cls, node, **kwargs):
        return cls._common(node, **kwargs)
