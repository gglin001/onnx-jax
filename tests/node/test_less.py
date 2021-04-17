import numpy as np
import onnx

from tests.tools import expect


class Less:
    @staticmethod
    def export():  # type: () -> None
        node = onnx.helper.make_node(
            'Less',
            inputs=['x', 'y'],
            outputs=['less'],
        )

        x = np.random.randn(3, 4, 5).astype(np.float32)
        y = np.random.randn(3, 4, 5).astype(np.float32)
        z = np.less(x, y)
        expect(node, inputs=[x, y], outputs=[z], name='test_less')

    @staticmethod
    def export_less_broadcast():  # type: () -> None
        node = onnx.helper.make_node(
            'Less',
            inputs=['x', 'y'],
            outputs=['less'],
        )

        x = np.random.randn(3, 4, 5).astype(np.float32)
        y = np.random.randn(5).astype(np.float32)
        z = np.less(x, y)
        expect(node, inputs=[x, y], outputs=[z], name='test_less_bcast')


if __name__ == '__main__':
    Less.export()
    Less.export_less_broadcast()
