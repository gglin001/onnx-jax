import numpy as np
import onnx

from tests.tools import expect


class Cosh:
    @staticmethod
    def export():  # type: () -> None
        node = onnx.helper.make_node(
            'Cosh',
            inputs=['x'],
            outputs=['y'],
        )

        x = np.array([-1, 0, 1]).astype(np.float32)
        y = np.cosh(x)  # expected output [1.54308069,  1.,  1.54308069]
        expect(node, inputs=[x], outputs=[y], name='test_cosh_example')

        x = np.random.randn(3, 4, 5).astype(np.float32)
        y = np.cosh(x)
        expect(node, inputs=[x], outputs=[y], name='test_cosh')


if __name__ == '__main__':
    Cosh.export()
