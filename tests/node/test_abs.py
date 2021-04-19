import numpy as np
import onnx

from tests.tools import expect


class Abs:
    @staticmethod
    def export():  # type: () -> None
        node = onnx.helper.make_node(
            'Abs',
            inputs=['x'],
            outputs=['y'],
        )
        x = np.random.randn(3, 4, 5).astype(np.float32)
        y = abs(x)

        expect(node, inputs=[x], outputs=[y], name='test_abs')


if __name__ == '__main__':
    Abs.export()
