import jax.numpy as jnp
import numpy as np
import onnx
from onnx import defs, helper
from onnx.helper import make_opsetid
from onnx_jax.backend import run_node
from tests.tools import cosin_sim, expect, gen_random


class Acos:
    @staticmethod
    def export():  # type: () -> None
        node = onnx.helper.make_node(
            'Acos',
            inputs=['x'],
            outputs=['y'],
        )

        x = np.array([-0.5, 0, 0.5]).astype(np.float32)
        y = np.arccos(x)
        expect(node, inputs=[x], outputs=[y], name='test_acos_example')

        x = np.random.rand(3, 4, 5).astype(np.float32)
        y = np.arccos(x)
        expect(node, inputs=[x], outputs=[y], name='test_acos')


if __name__ == '__main__':
    Acos.export()
