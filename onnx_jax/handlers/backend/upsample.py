import jax.numpy as jnp
from jax.image import resize, ResizeMethod

from onnx_jax.handlers.backend_handler import BackendHandler
from onnx_jax.handlers.handler import onnx_op


@onnx_op("Upsample")
class Upsample(BackendHandler):
    @classmethod
    def _common(cls, node, inputs, **kwargs):
        return onnx_upsample_jax(*inputs, **node.attrs)

    @classmethod
    def version_7(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def version_9(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def version_10(cls, node, **kwargs):
        return cls._common(node, **kwargs)


def onnx_upsample_jax(x, scales=None, mode='nearest', **kwargs):

    sizes = jnp.asarray(x.shape) * scales
    sizes = sizes.astype(jnp.int32)

    if mode == 'nearest':
        method = ResizeMethod.NEAREST
    elif mode == 'linear':
        method = ResizeMethod.LINEAR
    elif mode == 'cubic':
        method = ResizeMethod.CUBIC
    else:
        raise NotImplemented(f"Resize mode: {mode}")

    return [resize(image=x, shape=sizes, method=method, antialias=True)]
