import numpy as np
import onnx

from tests.tools import expect


class Acosh:
    @staticmethod
    def export():  # type: () -> None
        node = onnx.helper.make_node(
            'Acosh',
            inputs=['x'],
            outputs=['y'],
        )

        x = np.array([10, np.e, 1]).astype(np.float32)
        y = np.arccosh(x)  # expected output [2.99322295,  1.65745449,  0.]
        expect(node, inputs=[x], outputs=[y], name='test_acosh_example')

        x = np.random.uniform(1.0, 10.0, (3, 4, 5)).astype(np.float32)
        y = np.arccosh(x)
        expect(node, inputs=[x], outputs=[y], name='test_acosh')


if __name__ == '__main__':
    Acosh.export()
