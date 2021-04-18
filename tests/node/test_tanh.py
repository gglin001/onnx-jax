import numpy as np
import onnx

from tests.tools import expect


class Tanh:
    @staticmethod
    def export():  # type: () -> None
        node = onnx.helper.make_node(
            'Tanh',
            inputs=['x'],
            outputs=['y'],
        )

        x = np.array([-1, 0, 1]).astype(np.float32)
        y = np.tanh(x)  # expected output [-0.76159418, 0., 0.76159418]
        expect(node, inputs=[x], outputs=[y], name='test_tanh_example')

        x = np.random.randn(3, 4, 5).astype(np.float32)
        y = np.tanh(x)
        expect(node, inputs=[x], outputs=[y], name='test_tanh')


if __name__ == '__main__':
    Tanh.export()
