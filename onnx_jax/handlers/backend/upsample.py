import inspect

import jax.numpy as jnp
from jax.image import ResizeMethod, resize

from onnx_jax.handlers.backend_handler import BackendHandler
from onnx_jax.handlers.handler import onnx_op
from onnx_jax.pb_wrapper import OnnxNode


@onnx_op("Upsample")
class Upsample(BackendHandler):
    @classmethod
    def _common(cls, node: OnnxNode, **kwargs):
        cls._rewrite(node)
        cls._prepare(node)

        return onnx_upsample

    @classmethod
    def version_7(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def version_9(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def version_10(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def _rewrite(cls, node: OnnxNode):
        if 'mode' not in node.attrs:
            node.attrs['mode'] = 'nearest'

    @classmethod
    def _prepare(cls, node: OnnxNode):
        args = list(inspect.signature(onnx_upsample).parameters.keys())
        attrs = [node.attrs.get(k, None) for k in args[node.len_inputs :]]
        node.attrs_list.extend(attrs)


# TODO jit
def onnx_upsample(x, scales: float, mode='nearest'):
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

    return resize(image=x, shape=sizes, method=method, antialias=True)
