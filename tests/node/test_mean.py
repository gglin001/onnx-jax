import numpy as np
import onnx

from tests.tools import expect


class Mean:
    @staticmethod
    def export():  # type: () -> None
        data_0 = np.array([3, 0, 2]).astype(np.float32)
        data_1 = np.array([1, 3, 4]).astype(np.float32)
        data_2 = np.array([2, 6, 6]).astype(np.float32)
        result = np.array([2, 3, 4]).astype(np.float32)
        node = onnx.helper.make_node(
            'Mean',
            inputs=['data_0', 'data_1', 'data_2'],
            outputs=['result'],
        )
        expect(
            node,
            inputs=[data_0, data_1, data_2],
            outputs=[result],
            name='test_mean_example',
        )

        node = onnx.helper.make_node(
            'Mean',
            inputs=['data_0'],
            outputs=['result'],
        )
        expect(node, inputs=[data_0], outputs=[data_0], name='test_mean_one_input')

        result = np.divide(np.add(data_0, data_1), 2.0)
        node = onnx.helper.make_node(
            'Mean',
            inputs=['data_0', 'data_1'],
            outputs=['result'],
        )
        expect(
            node, inputs=[data_0, data_1], outputs=[result], name='test_mean_two_inputs'
        )


if __name__ == '__main__':
    Mean.export()
