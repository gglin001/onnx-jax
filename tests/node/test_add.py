import numpy as np
import onnx
from tests.tools import expect


class Add:
    @staticmethod
    def export():  # type: () -> None
        node = onnx.helper.make_node(
            'Add',
            inputs=['x', 'y'],
            outputs=['sum'],
        )

        x = np.random.randn(3, 4, 5).astype(np.float32)
        y = np.random.randn(3, 4, 5).astype(np.float32)
        expect(node, inputs=[x, y], outputs=[x + y], name='test_add')

    @staticmethod
    def export_add_broadcast():  # type: () -> None
        node = onnx.helper.make_node(
            'Add',
            inputs=['x', 'y'],
            outputs=['sum'],
        )

        x = np.random.randn(3, 4, 5).astype(np.float32)
        y = np.random.randn(5).astype(np.float32)
        expect(node, inputs=[x, y], outputs=[x + y], name='test_add_bcast')


if __name__ == '__main__':
    Add.export()
