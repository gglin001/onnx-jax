import numpy as np
import onnx

from tests.tools import expect


class Squeeze:
    @staticmethod
    def export_squeeze():  # type: () -> None
        node = onnx.helper.make_node(
            'Squeeze',
            inputs=['x', 'axes'],
            outputs=['y'],
        )
        x = np.random.randn(1, 3, 4, 5).astype(np.float32)
        axes = np.array([0], dtype=np.int64)
        y = np.squeeze(x, axis=0)

        expect(node, inputs=[x, axes], outputs=[y], name='test_squeeze')

    @staticmethod
    def export_squeeze_negative_axes():  # type: () -> None
        node = onnx.helper.make_node(
            'Squeeze',
            inputs=['x', 'axes'],
            outputs=['y'],
        )
        x = np.random.randn(1, 3, 1, 5).astype(np.float32)
        axes = np.array([-2], dtype=np.int64)
        y = np.squeeze(x, axis=-2)
        expect(node, inputs=[x, axes], outputs=[y], name='test_squeeze_negative_axes')


if __name__ == '__main__':
    Squeeze.export_squeeze()
    Squeeze.export_squeeze_negative_axes()
