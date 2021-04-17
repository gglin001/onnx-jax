import numpy as np
import onnx

from tests.tools import expect


class GlobalMaxPool:
    @staticmethod
    def export():  # type: () -> None

        node = onnx.helper.make_node(
            'GlobalMaxPool',
            inputs=['x'],
            outputs=['y'],
        )
        x = np.random.randn(1, 3, 5, 5).astype(np.float32)
        spatial_shape = np.ndim(x) - 2
        y = np.max(x, axis=tuple(range(spatial_shape, spatial_shape + 2)))
        for _ in range(spatial_shape):
            y = np.expand_dims(y, -1)
        expect(node, inputs=[x], outputs=[y], name='test_globalmaxpool')

    @staticmethod
    def export_globalmaxpool_precomputed():  # type: () -> None

        node = onnx.helper.make_node(
            'GlobalMaxPool',
            inputs=['x'],
            outputs=['y'],
        )
        x = np.array(
            [
                [
                    [
                        [1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9],
                    ]
                ]
            ]
        ).astype(np.float32)
        y = np.array([[[[9]]]]).astype(np.float32)
        expect(node, inputs=[x], outputs=[y], name='test_globalmaxpool_precomputed')


if __name__ == '__main__':
    GlobalMaxPool.export()
    GlobalMaxPool.export_globalmaxpool_precomputed()
