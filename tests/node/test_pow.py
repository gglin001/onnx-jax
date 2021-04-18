import numpy as np
import onnx

from tests.tools import expect


def pow(x, y):  # type: ignore
    z = np.power(x, y).astype(x.dtype)
    return z


class Pow:
    @staticmethod
    def export():  # type: () -> None
        node = onnx.helper.make_node(
            'Pow',
            inputs=['x', 'y'],
            outputs=['z'],
        )

        x = np.array([1, 2, 3]).astype(np.float32)
        y = np.array([4, 5, 6]).astype(np.float32)
        z = pow(x, y)  # expected output [1., 32., 729.]
        expect(node, inputs=[x, y], outputs=[z], name='test_pow_example')

        x = np.arange(60).reshape(3, 4, 5).astype(np.float32)
        y = np.random.randn(3, 4, 5).astype(np.float32)
        z = pow(x, y)
        expect(node, inputs=[x, y], outputs=[z], name='test_pow')

    @staticmethod
    def export_pow_broadcast():  # type: () -> None
        node = onnx.helper.make_node(
            'Pow',
            inputs=['x', 'y'],
            outputs=['z'],
        )

        x = np.array([1, 2, 3]).astype(np.float32)
        y = np.array(2).astype(np.float32)
        z = pow(x, y)  # expected output [1., 4., 9.]
        expect(node, inputs=[x, y], outputs=[z], name='test_pow_bcast_scalar')

        node = onnx.helper.make_node(
            'Pow',
            inputs=['x', 'y'],
            outputs=['z'],
        )
        x = np.array([[1, 2, 3], [4, 5, 6]]).astype(np.float32)
        y = np.array([1, 2, 3]).astype(np.float32)
        # expected output [[1, 4, 27], [4, 25, 216]]
        z = pow(x, y)
        expect(node, inputs=[x, y], outputs=[z], name='test_pow_bcast_array')

    @staticmethod
    def export_types():  # type: () -> None
        node = onnx.helper.make_node(
            'Pow',
            inputs=['x', 'y'],
            outputs=['z'],
        )

        x = np.array([1, 2, 3]).astype(np.float32)
        y = np.array([4, 5, 6]).astype(np.int64)
        z = pow(x, y)  # expected output [1., 32., 729.]
        expect(node, inputs=[x, y], outputs=[z], name='test_pow_types_float32_int64')

        x = np.array([1, 2, 3]).astype(np.int64)
        y = np.array([4, 5, 6]).astype(np.float32)
        z = pow(x, y)  # expected output [1, 32, 729]
        expect(node, inputs=[x, y], outputs=[z], name='test_pow_types_int64_float32')

        x = np.array([1, 2, 3]).astype(np.float32)
        y = np.array([4, 5, 6]).astype(np.int32)
        z = pow(x, y)  # expected output [1., 32., 729.]
        expect(node, inputs=[x, y], outputs=[z], name='test_pow_types_float32_int32')

        x = np.array([1, 2, 3]).astype(np.int32)
        y = np.array([4, 5, 6]).astype(np.float32)
        z = pow(x, y)  # expected output [1, 32, 729]
        expect(node, inputs=[x, y], outputs=[z], name='test_pow_types_int32_float32')

        x = np.array([1, 2, 3]).astype(np.float32)
        y = np.array([4, 5, 6]).astype(np.uint64)
        z = pow(x, y)  # expected output [1., 32., 729.]
        expect(node, inputs=[x, y], outputs=[z], name='test_pow_types_float32_uint64')

        x = np.array([1, 2, 3]).astype(np.float32)
        y = np.array([4, 5, 6]).astype(np.uint32)
        z = pow(x, y)  # expected output [1., 32., 729.]
        expect(node, inputs=[x, y], outputs=[z], name='test_pow_types_float32_uint32')

        x = np.array([1, 2, 3]).astype(np.int64)
        y = np.array([4, 5, 6]).astype(np.int64)
        z = pow(x, y)  # expected output [1, 32, 729]
        expect(node, inputs=[x, y], outputs=[z], name='test_pow_types_int64_int64')

        x = np.array([1, 2, 3]).astype(np.int32)
        y = np.array([4, 5, 6]).astype(np.int32)
        z = pow(x, y)  # expected output [1, 32, 729]
        expect(node, inputs=[x, y], outputs=[z], name='test_pow_types_int32_int32')


if __name__ == '__main__':
    Pow.export()
    Pow.export_pow_broadcast()
    Pow.export_types()
