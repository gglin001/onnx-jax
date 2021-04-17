import numpy as np
import onnx

from tests.tools import expect


class LeakyRelu:
    @staticmethod
    def export():  # type: () -> None
        node = onnx.helper.make_node(
            'LeakyRelu', inputs=['x'], outputs=['y'], alpha=0.1
        )

        x = np.array([-1, 0, 1]).astype(np.float32)
        # expected output [-0.1, 0., 1.]
        y = np.clip(x, 0, np.inf) + np.clip(x, -np.inf, 0) * 0.1
        expect(node, inputs=[x], outputs=[y], name='test_leakyrelu_example')

        x = np.random.randn(3, 4, 5).astype(np.float32)
        y = np.clip(x, 0, np.inf) + np.clip(x, -np.inf, 0) * 0.1
        expect(node, inputs=[x], outputs=[y], name='test_leakyrelu')

    @staticmethod
    def export_leakyrelu_default():  # type: () -> None
        default_alpha = 0.01
        node = onnx.helper.make_node(
            'LeakyRelu',
            inputs=['x'],
            outputs=['y'],
        )
        x = np.random.randn(3, 4, 5).astype(np.float32)
        y = np.clip(x, 0, np.inf) + np.clip(x, -np.inf, 0) * default_alpha
        expect(node, inputs=[x], outputs=[y], name='test_leakyrelu_default')


if __name__ == '__main__':
    LeakyRelu.export()
    LeakyRelu.export_leakyrelu_default()
