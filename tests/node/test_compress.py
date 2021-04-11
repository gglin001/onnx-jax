import numpy as np
import onnx

from tests.tools import expect


class Compress:
    @staticmethod
    def export_compress_0():  # type: () -> None
        node = onnx.helper.make_node(
            'Compress',
            inputs=['input', 'condition'],
            outputs=['output'],
            axis=0,
        )
        input = np.array([[1, 2], [3, 4], [5, 6]]).astype(np.float32)
        condition = np.array([0, 1, 1])
        output = np.compress(condition, input, axis=0)
        # print(output)
        # [[ 3.  4.]
        # [ 5.  6.]]

        expect(
            node,
            inputs=[input, condition.astype(np.bool)],
            outputs=[output],
            name='test_compress_0',
        )

    @staticmethod
    def export_compress_1():  # type: () -> None
        node = onnx.helper.make_node(
            'Compress',
            inputs=['input', 'condition'],
            outputs=['output'],
            axis=1,
        )
        input = np.array([[1, 2], [3, 4], [5, 6]]).astype(np.float32)
        condition = np.array([0, 1])
        output = np.compress(condition, input, axis=1)
        # print(output)
        # [[ 2.]
        # [ 4.]
        # [ 6.]]

        expect(
            node,
            inputs=[input, condition.astype(np.bool)],
            outputs=[output],
            name='test_compress_1',
        )

    @staticmethod
    def export_compress_default_axis():  # type: () -> None
        node = onnx.helper.make_node(
            'Compress',
            inputs=['input', 'condition'],
            outputs=['output'],
        )
        input = np.array([[1, 2], [3, 4], [5, 6]]).astype(np.float32)
        condition = np.array([0, 1, 0, 0, 1])
        output = np.compress(condition, input)
        # print(output)
        # [ 2., 5.]

        expect(
            node,
            inputs=[input, condition.astype(np.bool)],
            outputs=[output],
            name='test_compress_default_axis',
        )

    @staticmethod
    def export_compress_negative_axis():  # type: () -> None
        node = onnx.helper.make_node(
            'Compress',
            inputs=['input', 'condition'],
            outputs=['output'],
            axis=-1,
        )
        input = np.array([[1, 2], [3, 4], [5, 6]]).astype(np.float32)
        condition = np.array([0, 1])
        output = np.compress(condition, input, axis=-1)
        # print(output)
        # [[ 2.]
        # [ 4.]
        # [ 6.]]
        expect(
            node,
            inputs=[input, condition.astype(np.bool)],
            outputs=[output],
            name='test_compress_negative_axis',
        )


if __name__ == '__main__':
    Compress.export_compress_0()
    Compress.export_compress_1()
    Compress.export_compress_default_axis()
    Compress.export_compress_negative_axis()
