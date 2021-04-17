import numpy as np
import onnx

from tests.tools import expect


class Floor:
    @staticmethod
    def export():  # type: () -> None
        node = onnx.helper.make_node(
            'Floor',
            inputs=['x'],
            outputs=['y'],
        )

        x = np.array([-1.5, 1.2, 2]).astype(np.float32)
        y = np.floor(x)  # expected output [-2., 1., 2.]
        expect(node, inputs=[x], outputs=[y], name='test_floor_example')

        x = np.random.randn(3, 4, 5).astype(np.float32)
        y = np.floor(x)
        expect(node, inputs=[x], outputs=[y], name='test_floor')


if __name__ == '__main__':
    Floor.export()
