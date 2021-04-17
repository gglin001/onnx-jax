import numpy as np
import onnx

from tests.tools import expect


class Identity:
    @staticmethod
    def export():  # type: () -> None
        node = onnx.helper.make_node(
            'Identity',
            inputs=['x'],
            outputs=['y'],
        )

        data = np.array(
            [
                [
                    [
                        [1, 2],
                        [3, 4],
                    ]
                ]
            ],
            dtype=np.float32,
        )

        expect(node, inputs=[data], outputs=[data], name='test_identity')

    @staticmethod
    def export_sequence():  # type: () -> None
        node = onnx.helper.make_node(
            'Identity',
            inputs=['x'],
            outputs=['y'],
        )

        data = [
            np.array(
                [
                    [
                        [
                            [1, 2],
                            [3, 4],
                        ]
                    ]
                ],
                dtype=np.float32,
            ),
            np.array(
                [
                    [
                        [
                            [2, 3],
                            [1, 5],
                        ]
                    ]
                ],
                dtype=np.float32,
            ),
        ]

        expect(node, inputs=[data], outputs=[data], name='test_identity_sequence')


if __name__ == '__main__':
    Identity.export()
    # Identity.export_sequence() # pass
