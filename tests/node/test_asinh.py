import numpy as np
import onnx
from tests.tools import expect


class Asinh:
    @staticmethod
    def export():  # type: () -> None
        node = onnx.helper.make_node(
            'Asinh',
            inputs=['x'],
            outputs=['y'],
        )

        x = np.array([-1, 0, 1]).astype(np.float32)
        y = np.arcsinh(x)  # expected output [-0.88137358,  0.,  0.88137358]
        expect(node, inputs=[x], outputs=[y], name='test_asinh_example')

        x = np.random.randn(3, 4, 5).astype(np.float32)
        y = np.arcsinh(x)
        expect(node, inputs=[x], outputs=[y], name='test_asinh')


if __name__ == '__main__':
    Asinh.export()
