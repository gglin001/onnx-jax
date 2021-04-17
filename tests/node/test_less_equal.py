import numpy as np
import onnx

from tests.tools import expect


class LessOrEqual:
    @staticmethod
    def export():  # type: () -> None
        node = onnx.helper.make_node(
            'LessOrEqual',
            inputs=['x', 'y'],
            outputs=['less_equal'],
        )

        x = np.random.randn(3, 4, 5).astype(np.float32)
        y = np.random.randn(3, 4, 5).astype(np.float32)
        z = np.less_equal(x, y)
        expect(node, inputs=[x, y], outputs=[z], name='test_less_equal')

    @staticmethod
    def export_less_broadcast():  # type: () -> None
        node = onnx.helper.make_node(
            'LessOrEqual',
            inputs=['x', 'y'],
            outputs=['less_equal'],
        )

        x = np.random.randn(3, 4, 5).astype(np.float32)
        y = np.random.randn(5).astype(np.float32)
        z = np.less_equal(x, y)
        expect(node, inputs=[x, y], outputs=[z], name='test_less_equal_bcast')


if __name__ == '__main__':
    LessOrEqual.export()
    LessOrEqual.export_less_broadcast()
