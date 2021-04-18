import numpy as np
import onnx

from tests.tools import expect


class Sinh:
    @staticmethod
    def export():  # type: () -> None
        node = onnx.helper.make_node(
            'Sinh',
            inputs=['x'],
            outputs=['y'],
        )

        x = np.array([-1, 0, 1]).astype(np.float32)
        y = np.sinh(x)  # expected output [-1.17520118,  0.,  1.17520118]
        expect(node, inputs=[x], outputs=[y], name='test_sinh_example')

        x = np.random.randn(3, 4, 5).astype(np.float32)
        y = np.sinh(x)
        expect(node, inputs=[x], outputs=[y], name='test_sinh')


if __name__ == '__main__':
    Sinh.export()
