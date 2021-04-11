import inspect
from functools import partial

import jax.numpy as jnp
from jax import jit
from onnx_jax.handlers.backend_handler import BackendHandler
from onnx_jax.handlers.handler import onnx_op
from onnx_jax.pb_wrapper import OnnxNode


@onnx_op("Reshape")
class Reshape(BackendHandler):
    @classmethod
    def _common(cls, node, **kwargs):
        cls._prepare(node)

        def _reshape(x, _shape):
            shape = list(_shape)
            new_shape = []
            for idx in range(len(shape)):
                if shape[idx] == 0:
                    new_shape.append(x.shape[idx])
                else:
                    new_shape.append(shape[idx])
            return reshape(x, tuple(new_shape))

        return _reshape

    @classmethod
    def version_1(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def version_5(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def version_13(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def _prepare(cls, node: OnnxNode):
        args = list(inspect.signature(reshape).parameters.keys())
        attrs = [node.attrs.get(k, None) for k in args[node.len_inputs :]]
        node.attrs_list.extend(attrs)


@partial(jit, static_argnums=(1))
def reshape(x, shape):
    return jnp.reshape(x, shape)
