import numpy as np
import onnx

from tests.tools import expect


class IsNaN:
    @staticmethod
    def export():  # type: () -> None
        node = onnx.helper.make_node(
            'IsNaN',
            inputs=['x'],
            outputs=['y'],
        )

        x = np.array([3.0, np.nan, 4.0, np.nan], dtype=np.float32)
        y = np.isnan(x)
        expect(node, inputs=[x], outputs=[y], name='test_isnan')


if __name__ == '__main__':
    IsNaN.export()
