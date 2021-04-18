import itertools

import numpy as np
import onnx

from tests.tools import expect


class Transpose:
    @staticmethod
    def export_default():  # type: () -> None
        shape = (2, 3, 4)
        data = np.random.random_sample(shape).astype(np.float32)

        node = onnx.helper.make_node(
            'Transpose', inputs=['data'], outputs=['transposed']
        )

        transposed = np.transpose(data)
        expect(node, inputs=[data], outputs=[transposed], name='test_transpose_default')

    @staticmethod
    def export_all_permutations():  # type: () -> None
        shape = (2, 3, 4)
        data = np.random.random_sample(shape).astype(np.float32)
        permutations = list(itertools.permutations(np.arange(len(shape))))

        for i in range(len(permutations)):
            node = onnx.helper.make_node(
                'Transpose',
                inputs=['data'],
                outputs=['transposed'],
                perm=permutations[i],
            )
            transposed = np.transpose(data, permutations[i])
            expect(
                node,
                inputs=[data],
                outputs=[transposed],
                name='test_transpose_all_permutations_' + str(i),
            )


if __name__ == '__main__':
    Transpose.export_all_permutations()
    Transpose.export_default()
