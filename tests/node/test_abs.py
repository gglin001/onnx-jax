import jax.numpy as jnp
from onnx import defs, helper
from onnx.helper import make_opsetid
from onnx_jax.backend import run_node
from tests.tools import cosin_sim, gen_random


def test_abs():
    node_def = helper.make_node("Abs", ["X"], ["Y"])
    x = gen_random(shape=[1000])

    opset = make_opsetid(defs.ONNX_DOMAIN, 1)
    output = run_node(node_def, [x], opset=[opset])
    sim = cosin_sim(output[0], jnp.abs(x))
    print(sim)


if __name__ == '__main__':
    test_abs()
