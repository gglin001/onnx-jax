import numpy as np
import onnx

from tests.tools import expect


class ReduceProd:
    @staticmethod
    def export_do_not_keepdims():  # type: () -> None
        shape = [3, 2, 2]
        axes = [1]
        keepdims = 0

        node = onnx.helper.make_node(
            'ReduceProd',
            inputs=['data'],
            outputs=['reduced'],
            axes=axes,
            keepdims=keepdims,
        )

        data = np.array(
            [[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]], dtype=np.float32
        )
        reduced = np.prod(data, axis=tuple(axes), keepdims=keepdims == 1)
        # print(reduced)
        # [[3., 8.]
        # [35., 48.]
        # [99., 120.]]

        expect(
            node,
            inputs=[data],
            outputs=[reduced],
            name='test_reduce_prod_do_not_keepdims_example',
        )

        np.random.seed(0)
        data = np.random.uniform(-10, 10, shape).astype(np.float32)
        reduced = np.prod(data, axis=tuple(axes), keepdims=keepdims == 1)
        expect(
            node,
            inputs=[data],
            outputs=[reduced],
            name='test_reduce_prod_do_not_keepdims_random',
        )

    @staticmethod
    def export_keepdims():  # type: () -> None
        shape = [3, 2, 2]
        axes = [1]
        keepdims = 1

        node = onnx.helper.make_node(
            'ReduceProd',
            inputs=['data'],
            outputs=['reduced'],
            axes=axes,
            keepdims=keepdims,
        )

        data = np.array(
            [[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]], dtype=np.float32
        )
        reduced = np.prod(data, axis=tuple(axes), keepdims=keepdims == 1)
        # print(reduced)
        # [[[3., 8.]]
        # [[35., 48.]]
        # [[99., 120.]]]

        expect(
            node,
            inputs=[data],
            outputs=[reduced],
            name='test_reduce_prod_keepdims_example',
        )

        np.random.seed(0)
        data = np.random.uniform(-10, 10, shape).astype(np.float32)
        reduced = np.prod(data, axis=tuple(axes), keepdims=keepdims == 1)
        expect(
            node,
            inputs=[data],
            outputs=[reduced],
            name='test_reduce_prod_keepdims_random',
        )

    @staticmethod
    def export_default_axes_keepdims():  # type: () -> None
        shape = [3, 2, 2]
        axes = None
        keepdims = 1

        node = onnx.helper.make_node(
            'ReduceProd', inputs=['data'], outputs=['reduced'], keepdims=keepdims
        )

        data = np.array(
            [[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]], dtype=np.float32
        )
        reduced = np.prod(data, axis=axes, keepdims=keepdims == 1)
        # print(reduced)
        # [[[4.790016e+08]]]

        expect(
            node,
            inputs=[data],
            outputs=[reduced],
            name='test_reduce_prod_default_axes_keepdims_example',
        )

        np.random.seed(0)
        data = np.random.uniform(-10, 10, shape).astype(np.float32)
        reduced = np.prod(data, axis=axes, keepdims=keepdims == 1)
        expect(
            node,
            inputs=[data],
            outputs=[reduced],
            name='test_reduce_prod_default_axes_keepdims_random',
        )

    @staticmethod
    def export_negative_axes_keepdims():  # type: () -> None
        shape = [3, 2, 2]
        axes = [-2]
        keepdims = 1

        node = onnx.helper.make_node(
            'ReduceProd',
            inputs=['data'],
            outputs=['reduced'],
            axes=axes,
            keepdims=keepdims,
        )

        data = np.array(
            [[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]], dtype=np.float32
        )
        reduced = np.prod(data, axis=tuple(axes), keepdims=keepdims == 1)
        # print(reduced)
        # [[[3., 8.]]
        # [[35., 48.]]
        # [[99., 120.]]]

        expect(
            node,
            inputs=[data],
            outputs=[reduced],
            name='test_reduce_prod_negative_axes_keepdims_example',
        )

        np.random.seed(0)
        data = np.random.uniform(-10, 10, shape).astype(np.float32)
        reduced = np.prod(data, axis=tuple(axes), keepdims=keepdims == 1)
        expect(
            node,
            inputs=[data],
            outputs=[reduced],
            name='test_reduce_prod_negative_axes_keepdims_random',
        )


if __name__ == '__main__':
    ReduceProd.export_default_axes_keepdims()
    ReduceProd.export_do_not_keepdims()
    ReduceProd.export_keepdims()
    ReduceProd.export_negative_axes_keepdims()
