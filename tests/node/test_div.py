import numpy as np
import onnx

from tests.tools import expect


class Div:
    @staticmethod
    def export():  # type: () -> None
        node = onnx.helper.make_node(
            'Div',
            inputs=['x', 'y'],
            outputs=['z'],
        )

        x = np.array([3, 4]).astype(np.float32)
        y = np.array([1, 2]).astype(np.float32)
        z = x / y  # expected output [3., 2.]
        expect(node, inputs=[x, y], outputs=[z], name='test_div_example')

        x = np.random.randn(3, 4, 5).astype(np.float32)
        y = np.random.rand(3, 4, 5).astype(np.float32) + 1.0
        z = x / y
        expect(node, inputs=[x, y], outputs=[z], name='test_div')

    @staticmethod
    def export_div_broadcast():  # type: () -> None
        node = onnx.helper.make_node(
            'Div',
            inputs=['x', 'y'],
            outputs=['z'],
        )

        x = np.random.randn(3, 4, 5).astype(np.float32)
        y = np.random.rand(5).astype(np.float32) + 1.0
        z = x / y
        expect(node, inputs=[x, y], outputs=[z], name='test_div_bcast')


if __name__ == "__main__":
    Div.export()
    Div.export_div_broadcast()
