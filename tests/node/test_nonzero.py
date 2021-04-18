import numpy as np
import onnx

from tests.tools import expect


class NonZero:
    @staticmethod
    def export():  # type: () -> None
        node = onnx.helper.make_node(
            'NonZero',
            inputs=['condition'],
            outputs=['result'],
        )

        condition = np.array([[1, 0], [1, 1]], dtype=np.bool)
        result = np.array(
            np.nonzero(condition), dtype=np.int64
        )  # expected output [[0, 1, 1], [0, 0, 1]]
        expect(node, inputs=[condition], outputs=[result], name='test_nonzero_example')


if __name__ == '__main__':
    NonZero.export()
