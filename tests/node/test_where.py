import numpy as np
import onnx

from tests.tools import expect


class Where:
    @staticmethod
    def export():  # type: () -> None
        node = onnx.helper.make_node(
            'Where',
            inputs=['condition', 'x', 'y'],
            outputs=['z'],
        )

        condition = np.array([[1, 0], [1, 1]], dtype=np.bool)
        x = np.array([[1, 2], [3, 4]], dtype=np.float32)
        y = np.array([[9, 8], [7, 6]], dtype=np.float32)
        z = np.where(condition, x, y)  # expected output [[1, 8], [3, 4]]
        expect(node, inputs=[condition, x, y], outputs=[z], name='test_where_example')

    @staticmethod
    def export_long():  # type: () -> None
        node = onnx.helper.make_node(
            'Where',
            inputs=['condition', 'x', 'y'],
            outputs=['z'],
        )

        condition = np.array([[1, 0], [1, 1]], dtype=np.bool)
        x = np.array([[1, 2], [3, 4]], dtype=np.int64)
        y = np.array([[9, 8], [7, 6]], dtype=np.int64)
        z = np.where(condition, x, y)  # expected output [[1, 8], [3, 4]]
        expect(
            node, inputs=[condition, x, y], outputs=[z], name='test_where_long_example'
        )


if __name__ == '__main__':
    Where.export()
    Where.export_long()
