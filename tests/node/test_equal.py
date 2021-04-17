import numpy as np
import onnx

from tests.tools import expect


class Equal:
    @staticmethod
    def export():  # type: () -> None
        node = onnx.helper.make_node(
            'Equal',
            inputs=['x', 'y'],
            outputs=['z'],
        )

        x = (np.random.randn(3, 4, 5) * 10).astype(np.int32)
        y = (np.random.randn(3, 4, 5) * 10).astype(np.int32)
        z = np.equal(x, y)
        expect(node, inputs=[x, y], outputs=[z], name='test_equal')

    @staticmethod
    def export_equal_broadcast():  # type: () -> None
        node = onnx.helper.make_node(
            'Equal',
            inputs=['x', 'y'],
            outputs=['z'],
        )

        x = (np.random.randn(3, 4, 5) * 10).astype(np.int32)
        y = (np.random.randn(5) * 10).astype(np.int32)
        z = np.equal(x, y)
        expect(node, inputs=[x, y], outputs=[z], name='test_equal_bcast')


if __name__ == "__main__":
    Equal.export()
    Equal.export_equal_broadcast()
