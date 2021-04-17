import math

import numpy as np
import onnx

from tests.tools import expect


class Erf:
    @staticmethod
    def export():  # type: () -> None
        node = onnx.helper.make_node(
            'Erf',
            inputs=['x'],
            outputs=['y'],
        )

        x = np.random.randn(1, 3, 32, 32).astype(np.float32)
        y = np.vectorize(math.erf)(x).astype(np.float32)
        expect(node, inputs=[x], outputs=[y], name='test_erf')


if __name__ == '__main__':
    Erf.export()
