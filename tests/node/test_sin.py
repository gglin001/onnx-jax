import numpy as np
import onnx

from tests.tools import expect


class Sin:
    @staticmethod
    def export():  # type: () -> None
        node = onnx.helper.make_node(
            'Sin',
            inputs=['x'],
            outputs=['y'],
        )

        x = np.array([-1, 0, 1]).astype(np.float32)
        y = np.sin(x)
        expect(node, inputs=[x], outputs=[y], name='test_sin_example')

        x = np.random.randn(3, 4, 5).astype(np.float32)
        y = np.sin(x)
        expect(node, inputs=[x], outputs=[y], name='test_sin')


if __name__ == '__main__':
    Sin.export()
