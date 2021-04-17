import numpy as np
import onnx

from tests.tools import expect


class Cos:
    @staticmethod
    def export():  # type: () -> None
        node = onnx.helper.make_node(
            'Cos',
            inputs=['x'],
            outputs=['y'],
        )

        x = np.array([-1, 0, 1]).astype(np.float32)
        y = np.cos(x)
        expect(node, inputs=[x], outputs=[y], name='test_cos_example')

        x = np.random.randn(3, 4, 5).astype(np.float32)
        y = np.cos(x)
        expect(node, inputs=[x], outputs=[y], name='test_cos')


if __name__ == '__main__':
    Cos.export()
