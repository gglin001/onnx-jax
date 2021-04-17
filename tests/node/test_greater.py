import numpy as np
import onnx

from tests.tools import expect


class Greater:
    @staticmethod
    def export():  # type: () -> None
        node = onnx.helper.make_node(
            'Greater',
            inputs=['x', 'y'],
            outputs=['greater'],
        )

        x = np.random.randn(3, 4, 5).astype(np.float32)
        y = np.random.randn(3, 4, 5).astype(np.float32)
        z = np.greater(x, y)
        expect(node, inputs=[x, y], outputs=[z], name='test_greater')

    @staticmethod
    def export_greater_broadcast():  # type: () -> None
        node = onnx.helper.make_node(
            'Greater',
            inputs=['x', 'y'],
            outputs=['greater'],
        )

        x = np.random.randn(3, 4, 5).astype(np.float32)
        y = np.random.randn(5).astype(np.float32)
        z = np.greater(x, y)
        expect(node, inputs=[x, y], outputs=[z], name='test_greater_bcast')


if __name__ == '__main__':
    Greater.export()
    Greater.export_greater_broadcast()
