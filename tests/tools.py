import jax
import jax.numpy as jnp

from onnx_jax.backend import run_node
from onnx_jax.logger import logger

random_key = jax.random.PRNGKey(0)


def gen_random(minval=-1.0, maxval=1.0, shape=None, dtype=jnp.float32):
    output = jax.random.uniform(
        random_key, shape=shape, dtype=dtype, minval=minval, maxval=maxval
    )
    return output


def cosin_sim(a, b):
    a = a.astype(jnp.float32)
    b = b.astype(jnp.float32)
    a = a.flatten()
    b = b.flatten()
    cos_sim = jnp.dot(a, b) / (jnp.linalg.norm(a) * jnp.linalg.norm(b))
    return cos_sim


def expect(node, inputs, outputs, **kwargs):
    outputs_jax = run_node(node, inputs)

    logger.info(f"golden size: {outputs[0].shape}, output size: {outputs_jax[0].shape}")
    if list(outputs[0].shape) == list(outputs_jax[0].shape):
        sim = cosin_sim(outputs_jax[0], jnp.asarray(outputs[0]))
        logger.info(f"\tcosin similarity: {sim}")
    else:
        logger.error('failed')
