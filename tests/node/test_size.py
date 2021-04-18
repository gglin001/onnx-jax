import numpy as np
import onnx

from tests.tools import expect


class Size:
    @staticmethod
    def export():  # type: () -> None
        node = onnx.helper.make_node(
            'Size',
            inputs=['x'],
            outputs=['y'],
        )

        x = np.array(
            [
                [1, 2, 3],
                [4, 5, 6],
            ]
        ).astype(np.float32)
        y = np.array(6).astype(np.int64)

        expect(node, inputs=[x], outputs=[y], name='test_size_example')

        x = np.random.randn(3, 4, 5).astype(np.float32)
        y = np.array(x.size).astype(np.int64)

        expect(node, inputs=[x], outputs=[y], name='test_size')


if __name__ == '__main__':
    Size.export()
