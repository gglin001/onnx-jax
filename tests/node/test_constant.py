import numpy as np
import onnx

from tests.tools import expect


class Constant:
    @staticmethod
    def export():  # type: () -> None
        values = np.random.randn(5, 5).astype(np.float32)
        node = onnx.helper.make_node(
            'Constant',
            inputs=[],
            outputs=['values'],
            value=onnx.helper.make_tensor(
                name='const_tensor',
                data_type=onnx.TensorProto.FLOAT,
                dims=values.shape,
                vals=values.flatten().astype(float),
            ),
        )

        expect(node, inputs=[], outputs=[values], name='test_constant')


if __name__ == '__main__':
    Constant.export()
