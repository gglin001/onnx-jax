import inspect
from functools import partial

import jax.numpy as jnp
from jax import jit

from onnx_jax.handlers.backend_handler import BackendHandler
from onnx_jax.handlers.handler import onnx_op
from onnx_jax.logger import logger
from onnx_jax.pb_wrapper import OnnxNode


@onnx_op("Dropout")
class Dropout(BackendHandler):
    @classmethod
    def _common(cls, node: OnnxNode, **kwargs):
        cls._rewrite(node)
        cls._prepare(node)

        def _dropout(x, ratio=None, training_mode=False, return_mask=False):
            return onnx_dropout(x, return_mask)

        return _dropout

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
    def version_10(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def version_12(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def version_13(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def _rewrite(cls, node: OnnxNode):
        training_mode = node.attrs.get('training_mode', False)
        if training_mode:
            raise NotImplemented('Dropout training mode')
        ratio = node.attrs.get('ratio', 0.0)
        if ratio != 0.0:
            logger.warning(f"Dropout, change ratio from {ratio:.4f} to 0.0")
        node.attrs['return_mask'] = True if node.len_outputs == 2 else False
        # ignore seed

    @classmethod
    def _prepare(cls, node: OnnxNode):
        args = list(inspect.signature(onnx_dropout).parameters.keys())
        attrs = [node.attrs.get(k, None) for k in args[node.len_inputs :]]
        node.attrs_list.extend(attrs)


@partial(jit, static_argnums=(1))
def onnx_dropout(x, return_mask=False):
    if return_mask:
        return (x, jnp.ones(x.shape, dtype=jnp.int32))
    else:
        return x
