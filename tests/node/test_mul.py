import numpy as np
import onnx

from tests.tools import expect


class Mul:
    @staticmethod
    def export():  # type: () -> None
        node = onnx.helper.make_node(
            'Mul',
            inputs=['x', 'y'],
            outputs=['z'],
        )

        x = np.array([1, 2, 3]).astype(np.float32)
        y = np.array([4, 5, 6]).astype(np.float32)
        z = x * y  # expected output [4., 10., 18.]
        expect(node, inputs=[x, y], outputs=[z], name='test_mul_example')

        x = np.random.randn(3, 4, 5).astype(np.float32)
        y = np.random.randn(3, 4, 5).astype(np.float32)
        z = x * y
        expect(node, inputs=[x, y], outputs=[z], name='test_mul')

        x = np.random.randint(4, size=(3, 4, 5), dtype=np.uint8)
        y = np.random.randint(24, size=(3, 4, 5), dtype=np.uint8)
        z = x * y
        expect(node, inputs=[x, y], outputs=[z], name='test_mul_uint8')

    @staticmethod
    def export_mul_broadcast():  # type: () -> None
        node = onnx.helper.make_node(
            'Mul',
            inputs=['x', 'y'],
            outputs=['z'],
        )

        x = np.random.randn(3, 4, 5).astype(np.float32)
        y = np.random.randn(5).astype(np.float32)
        z = x * y
        expect(node, inputs=[x, y], outputs=[z], name='test_mul_bcast')


if __name__ == '__main__':
    Mul.export()
    Mul.export_mul_broadcast()
