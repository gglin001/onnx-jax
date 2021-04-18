import numpy as np
import onnx

from tests.tools import expect


class ReduceLogSum:
    @staticmethod
    def export_nokeepdims():  # type: () -> None
        node = onnx.helper.make_node(
            'ReduceLogSum',
            inputs=['data'],
            outputs=["reduced"],
            axes=[2, 1],
            keepdims=0,
        )
        data = np.random.ranf([3, 4, 5]).astype(np.float32)
        reduced = np.log(np.sum(data, axis=(2, 1), keepdims=False))
        expect(
            node, inputs=[data], outputs=[reduced], name='test_reduce_log_sum_desc_axes'
        )

        node = onnx.helper.make_node(
            'ReduceLogSum',
            inputs=['data'],
            outputs=["reduced"],
            axes=[0, 1],
            keepdims=0,
        )
        data = np.random.ranf([3, 4, 5]).astype(np.float32)
        reduced = np.log(np.sum(data, axis=(0, 1), keepdims=False))
        expect(
            node, inputs=[data], outputs=[reduced], name='test_reduce_log_sum_asc_axes'
        )

    @staticmethod
    def export_keepdims():  # type: () -> None
        node = onnx.helper.make_node(
            'ReduceLogSum', inputs=['data'], outputs=["reduced"]
        )
        data = np.random.ranf([3, 4, 5]).astype(np.float32)
        reduced = np.log(np.sum(data, keepdims=True))
        expect(
            node, inputs=[data], outputs=[reduced], name='test_reduce_log_sum_default'
        )

    @staticmethod
    def export_negative_axes_keepdims():  # type: () -> None
        node = onnx.helper.make_node(
            'ReduceLogSum', inputs=['data'], outputs=["reduced"], axes=[-2]
        )
        data = np.random.ranf([3, 4, 5]).astype(np.float32)
        reduced = np.log(np.sum(data, axis=(-2), keepdims=True))
        # print(reduced)
        expect(
            node,
            inputs=[data],
            outputs=[reduced],
            name='test_reduce_log_sum_negative_axes',
        )


if __name__ == '__main__':
    ReduceLogSum.export_keepdims()
    ReduceLogSum.export_negative_axes_keepdims()
    ReduceLogSum.export_nokeepdims()
