import jax.numpy as jnp

from onnx_jax.handlers.backend_handler import BackendHandler
from onnx_jax.handlers.handler import onnx_op


@onnx_op("Gemm")
class Gemm(BackendHandler):
    @classmethod
    def _common(cls, node, inputs, **kwargs):
        return gemm_reference_implementation(*inputs, **node.attrs)

    @classmethod
    def version_1(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def version_6(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def version_7(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def version_9(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def version_11(cls, node, **kwargs):
        return cls._common(node, **kwargs)


def gemm_reference_implementation(
    A, B, C=None, alpha=1.0, beta=1.0, transA=0, transB=0
):
    A = A if transA == 0 else A.T
    B = B if transB == 0 else B.T
    C = C if C is not None else jnp.array(0)

    return [alpha * jnp.dot(A, B) + beta * C]
