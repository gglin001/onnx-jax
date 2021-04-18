import numpy as np
import onnx

from tests.tools import expect


class Neg:
    @staticmethod
    def export():  # type: () -> None
        node = onnx.helper.make_node(
            'Neg',
            inputs=['x'],
            outputs=['y'],
        )

        x = np.array([-4, 2]).astype(np.float32)
        y = np.negative(x)  # expected output [4., -2.],
        expect(node, inputs=[x], outputs=[y], name='test_neg_example')

        x = np.random.randn(3, 4, 5).astype(np.float32)
        y = np.negative(x)
        expect(node, inputs=[x], outputs=[y], name='test_neg')


if __name__ == '__main__':
    Neg.export()
