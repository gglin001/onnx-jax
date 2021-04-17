import numpy as np
import onnx

from tests.tools import expect


class Det:
    @staticmethod
    def export_2d():  # type: () -> None
        node = onnx.helper.make_node(
            'Det',
            inputs=['x'],
            outputs=['y'],
        )

        x = np.arange(4).reshape(2, 2).astype(np.float32)
        y = np.linalg.det(x)  # expect -2
        expect(node, inputs=[x], outputs=[y], name='test_det_2d')

    @staticmethod
    def export_nd():  # type: () -> None
        node = onnx.helper.make_node(
            'Det',
            inputs=['x'],
            outputs=['y'],
        )

        x = np.array([[[1, 2], [3, 4]], [[1, 2], [2, 1]], [[1, 3], [3, 1]]]).astype(
            np.float32
        )
        y = np.linalg.det(x)  # expect array([-2., -3., -8.])
        expect(node, inputs=[x], outputs=[y], name='test_det_nd')


if __name__ == '__main__':
    Det.export_2d()
    Det.export_nd()
