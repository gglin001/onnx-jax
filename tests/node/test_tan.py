import numpy as np
import onnx

from tests.tools import expect


class Tan:
    @staticmethod
    def export():  # type: () -> None
        node = onnx.helper.make_node(
            'Tan',
            inputs=['x'],
            outputs=['y'],
        )

        x = np.array([-1, 0, 1]).astype(np.float32)
        y = np.tan(x)
        expect(node, inputs=[x], outputs=[y], name='test_tan_example')

        x = np.random.randn(3, 4, 5).astype(np.float32)
        y = np.tan(x)
        expect(node, inputs=[x], outputs=[y], name='test_tan')


if __name__ == '__main__':
    Tan.export()
