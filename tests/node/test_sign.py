import numpy as np
import onnx

from tests.tools import expect


class Sign:
    @staticmethod
    def export():  # type: () -> None
        node = onnx.helper.make_node(
            'Sign',
            inputs=['x'],
            outputs=['y'],
        )

        x = np.array(range(-5, 6)).astype(np.float32)
        y = np.sign(x)
        expect(node, inputs=[x], outputs=[y], name='test_sign')


if __name__ == '__main__':
    Sign.export()
