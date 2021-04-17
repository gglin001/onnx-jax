import numpy as np
import onnx

from tests.tools import expect

all_numeric_dtypes = [
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
    np.float16,
    np.float32,
    np.float64,
]


class Max:
    @staticmethod
    def export():  # type: () -> None
        data_0 = np.array([3, 2, 1]).astype(np.float32)
        data_1 = np.array([1, 4, 4]).astype(np.float32)
        data_2 = np.array([2, 5, 3]).astype(np.float32)
        result = np.array([3, 5, 4]).astype(np.float32)
        node = onnx.helper.make_node(
            'Max',
            inputs=['data_0', 'data_1', 'data_2'],
            outputs=['result'],
        )
        expect(
            node,
            inputs=[data_0, data_1, data_2],
            outputs=[result],
            name='test_max_example',
        )

        node = onnx.helper.make_node(
            'Max',
            inputs=['data_0'],
            outputs=['result'],
        )
        expect(node, inputs=[data_0], outputs=[data_0], name='test_max_one_input')

        result = np.maximum(data_0, data_1)
        node = onnx.helper.make_node(
            'Max',
            inputs=['data_0', 'data_1'],
            outputs=['result'],
        )
        expect(
            node, inputs=[data_0, data_1], outputs=[result], name='test_max_two_inputs'
        )

    @staticmethod
    def export_max_all_numeric_types():  # type: () -> None
        for op_dtype in all_numeric_dtypes:
            data_0 = np.array([3, 2, 1]).astype(op_dtype)
            data_1 = np.array([1, 4, 4]).astype(op_dtype)
            result = np.array([3, 4, 4]).astype(op_dtype)
            node = onnx.helper.make_node(
                'Max',
                inputs=['data_0', 'data_1'],
                outputs=['result'],
            )
            expect(
                node,
                inputs=[data_0, data_1],
                outputs=[result],
                name='test_max_{0}'.format(np.dtype(op_dtype).name),
            )


if __name__ == '__main__':
    Max.export()
    Max.export_max_all_numeric_types()
