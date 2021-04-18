import numpy as np
import onnx

from tests.tools import expect


class Sqrt:
    @staticmethod
    def export():  # type: () -> None
        node = onnx.helper.make_node(
            'Sqrt',
            inputs=['x'],
            outputs=['y'],
        )

        x = np.array([1, 4, 9]).astype(np.float32)
        y = np.sqrt(x)  # expected output [1., 2., 3.]
        expect(node, inputs=[x], outputs=[y], name='test_sqrt_example')

        x = np.abs(np.random.randn(3, 4, 5).astype(np.float32))
        y = np.sqrt(x)
        expect(node, inputs=[x], outputs=[y], name='test_sqrt')


if __name__ == '__main__':
    Sqrt.export()
