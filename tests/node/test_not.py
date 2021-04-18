import numpy as np
import onnx

from tests.tools import expect


class Not:
    @staticmethod
    def export():  # type: () -> None
        node = onnx.helper.make_node(
            'Not',
            inputs=['x'],
            outputs=['not'],
        )

        # 2d
        x = (np.random.randn(3, 4) > 0).astype(np.bool)
        expect(node, inputs=[x], outputs=[np.logical_not(x)], name='test_not_2d')

        # 3d
        x = (np.random.randn(3, 4, 5) > 0).astype(np.bool)
        expect(node, inputs=[x], outputs=[np.logical_not(x)], name='test_not_3d')

        # 4d
        x = (np.random.randn(3, 4, 5, 6) > 0).astype(np.bool)
        expect(node, inputs=[x], outputs=[np.logical_not(x)], name='test_not_4d')


if __name__ == '__main__':
    Not.export()
