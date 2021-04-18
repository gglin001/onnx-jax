import numpy as np
import onnx

from tests.tools import expect


class Shape:
    @staticmethod
    def export():  # type: () -> None
        node = onnx.helper.make_node(
            'Shape',
            inputs=['x'],
            outputs=['y'],
        )

        x = np.array(
            [
                [1, 2, 3],
                [4, 5, 6],
            ]
        ).astype(np.float32)
        y = np.array(
            [
                2,
                3,
            ]
        ).astype(np.int64)

        expect(node, inputs=[x], outputs=[y], name='test_shape_example')

        x = np.random.randn(3, 4, 5).astype(np.float32)
        y = np.array(x.shape).astype(np.int64)

        expect(node, inputs=[x], outputs=[y], name='test_shape')


if __name__ == '__main__':
    Shape.export()
