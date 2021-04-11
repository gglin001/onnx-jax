import numpy as np
import onnx

from tests.tools import expect


class Asin:
    @staticmethod
    def export():  # type: () -> None
        node = onnx.helper.make_node(
            'Asin',
            inputs=['x'],
            outputs=['y'],
        )

        x = np.array([-0.5, 0, 0.5]).astype(np.float32)
        y = np.arcsin(x)
        expect(node, inputs=[x], outputs=[y], name='test_asin_example')

        x = np.random.rand(3, 4, 5).astype(np.float32)
        y = np.arcsin(x)
        expect(node, inputs=[x], outputs=[y], name='test_asin')


if __name__ == '__main__':
    Asin.export()
