import numpy as np
import onnx

from tests.tools import expect


class GreaterOrEqual:
    @staticmethod
    def export():  # type: () -> None
        node = onnx.helper.make_node(
            'GreaterOrEqual',
            inputs=['x', 'y'],
            outputs=['greater_equal'],
        )

        x = np.random.randn(3, 4, 5).astype(np.float32)
        y = np.random.randn(3, 4, 5).astype(np.float32)
        z = np.greater_equal(x, y)
        expect(node, inputs=[x, y], outputs=[z], name='test_greater_equal')

    @staticmethod
    def export_greater_broadcast():  # type: () -> None
        node = onnx.helper.make_node(
            'GreaterOrEqual',
            inputs=['x', 'y'],
            outputs=['greater_equal'],
        )

        x = np.random.randn(3, 4, 5).astype(np.float32)
        y = np.random.randn(5).astype(np.float32)
        z = np.greater_equal(x, y)
        expect(node, inputs=[x, y], outputs=[z], name='test_greater_equal_bcast')


if __name__ == '__main__':
    GreaterOrEqual.export()
    GreaterOrEqual.export_greater_broadcast()
