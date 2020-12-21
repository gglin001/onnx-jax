import jax
import jax.numpy as jnp

random_key = jax.random.PRNGKey(0)


def _gen_random(minval=-1.0, maxval=1.0, shape=None, dtype=jnp.float32):
    output = jax.random.uniform(random_key, shape=shape, dtype=dtype, minval=minval, maxval=maxval)
    return output


def _cosin_sim(a, b):
    a = a.flatten()
    b = b.flatten()
    cos_sim = jnp.dot(a, b) / (jnp.linalg.norm(a) * jnp.linalg.norm(b))
    return cos_sim
