import numpy as np
import onnx

from tests.tools import expect


class Unsqueeze:
    @staticmethod
    def export_unsqueeze_one_axis():  # type: () -> None
        x = np.random.randn(3, 4, 5).astype(np.float32)

        for i in range(x.ndim):
            axes = np.array([i]).astype(np.int64)
            node = onnx.helper.make_node(
                'Unsqueeze',
                inputs=['x', 'axes'],
                outputs=['y'],
            )
            y = np.expand_dims(x, axis=i)

            expect(
                node,
                inputs=[x, axes],
                outputs=[y],
                name='test_unsqueeze_axis_' + str(i),
            )

    @staticmethod
    def export_unsqueeze_two_axes():  # type: () -> None
        x = np.random.randn(3, 4, 5).astype(np.float32)
        axes = np.array([1, 4]).astype(np.int64)

        node = onnx.helper.make_node(
            'Unsqueeze',
            inputs=['x', 'axes'],
            outputs=['y'],
        )
        y = np.expand_dims(x, axis=1)
        y = np.expand_dims(y, axis=4)

        expect(node, inputs=[x, axes], outputs=[y], name='test_unsqueeze_two_axes')

    @staticmethod
    def export_unsqueeze_three_axes():  # type: () -> None
        x = np.random.randn(3, 4, 5).astype(np.float32)
        axes = np.array([2, 4, 5]).astype(np.int64)

        node = onnx.helper.make_node(
            'Unsqueeze',
            inputs=['x', 'axes'],
            outputs=['y'],
        )
        y = np.expand_dims(x, axis=2)
        y = np.expand_dims(y, axis=4)
        y = np.expand_dims(y, axis=5)

        expect(node, inputs=[x, axes], outputs=[y], name='test_unsqueeze_three_axes')

    @staticmethod
    def export_unsqueeze_unsorted_axes():  # type: () -> None
        x = np.random.randn(3, 4, 5).astype(np.float32)
        axes = np.array([5, 4, 2]).astype(np.int64)

        node = onnx.helper.make_node(
            'Unsqueeze',
            inputs=['x', 'axes'],
            outputs=['y'],
        )
        y = np.expand_dims(x, axis=2)
        y = np.expand_dims(y, axis=4)
        y = np.expand_dims(y, axis=5)

        expect(node, inputs=[x, axes], outputs=[y], name='test_unsqueeze_unsorted_axes')

    @staticmethod
    def export_unsqueeze_negative_axes():  # type: () -> None
        node = onnx.helper.make_node(
            'Unsqueeze',
            inputs=['x', 'axes'],
            outputs=['y'],
        )
        x = np.random.randn(1, 3, 1, 5).astype(np.float32)
        axes = np.array([-2]).astype(np.int64)
        y = np.expand_dims(x, axis=-2)
        expect(node, inputs=[x, axes], outputs=[y], name='test_unsqueeze_negative_axes')


if __name__ == '__main__':
    Unsqueeze.export_unsqueeze_negative_axes()
    Unsqueeze.export_unsqueeze_one_axis()
    Unsqueeze.export_unsqueeze_three_axes()
    Unsqueeze.export_unsqueeze_two_axes()
    Unsqueeze.export_unsqueeze_unsorted_axes()
