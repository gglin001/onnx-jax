import numpy as np
import onnx

from tests.tools import expect


class Log:
    @staticmethod
    def export():  # type: () -> None
        node = onnx.helper.make_node(
            'Log',
            inputs=['x'],
            outputs=['y'],
        )

        x = np.array([1, 10]).astype(np.float32)
        y = np.log(x)  # expected output [0., 2.30258512]
        expect(node, inputs=[x], outputs=[y], name='test_log_example')

        x = np.exp(np.random.randn(3, 4, 5).astype(np.float32))
        y = np.log(x)
        expect(node, inputs=[x], outputs=[y], name='test_log')


if __name__ == '__main__':
    Log.export()
