import numpy as np
import onnx

from tests.tools import expect


class BitShift:
    @staticmethod
    def export_right_unit8():  # type: () -> None
        node = onnx.helper.make_node(
            'BitShift', inputs=['x', 'y'], outputs=['z'], direction="RIGHT"
        )

        x = np.array([16, 4, 1]).astype(np.uint8)
        y = np.array([1, 2, 3]).astype(np.uint8)
        z = x >> y  # expected output [8, 1, 0]
        expect(node, inputs=[x, y], outputs=[z], name='test_bitshift_right_uint8')

    @staticmethod
    def export_right_unit16():  # type: () -> None
        node = onnx.helper.make_node(
            'BitShift', inputs=['x', 'y'], outputs=['z'], direction="RIGHT"
        )

        x = np.array([16, 4, 1]).astype(np.uint16)
        y = np.array([1, 2, 3]).astype(np.uint16)
        z = x >> y  # expected output [8, 1, 0]
        expect(node, inputs=[x, y], outputs=[z], name='test_bitshift_right_uint16')

    @staticmethod
    def export_right_unit32():  # type: () -> None
        node = onnx.helper.make_node(
            'BitShift', inputs=['x', 'y'], outputs=['z'], direction="RIGHT"
        )

        x = np.array([16, 4, 1]).astype(np.uint32)
        y = np.array([1, 2, 3]).astype(np.uint32)
        z = x >> y  # expected output [8, 1, 0]
        expect(node, inputs=[x, y], outputs=[z], name='test_bitshift_right_uint32')

    @staticmethod
    def export_right_unit64():  # type: () -> None
        node = onnx.helper.make_node(
            'BitShift', inputs=['x', 'y'], outputs=['z'], direction="RIGHT"
        )

        x = np.array([16, 4, 1]).astype(np.uint64)
        y = np.array([1, 2, 3]).astype(np.uint64)
        z = x >> y  # expected output [8, 1, 0]
        expect(node, inputs=[x, y], outputs=[z], name='test_bitshift_right_uint64')

    @staticmethod
    def export_left_unit8():  # type: () -> None
        node = onnx.helper.make_node(
            'BitShift', inputs=['x', 'y'], outputs=['z'], direction="LEFT"
        )

        x = np.array([16, 4, 1]).astype(np.uint8)
        y = np.array([1, 2, 3]).astype(np.uint8)
        z = x << y  # expected output [32, 16, 8]
        expect(node, inputs=[x, y], outputs=[z], name='test_bitshift_left_uint8')

    @staticmethod
    def export_left_unit16():  # type: () -> None
        node = onnx.helper.make_node(
            'BitShift', inputs=['x', 'y'], outputs=['z'], direction="LEFT"
        )

        x = np.array([16, 4, 1]).astype(np.uint16)
        y = np.array([1, 2, 3]).astype(np.uint16)
        z = x << y  # expected output [32, 16, 8]
        expect(node, inputs=[x, y], outputs=[z], name='test_bitshift_left_uint16')

    @staticmethod
    def export_left_unit32():  # type: () -> None
        node = onnx.helper.make_node(
            'BitShift', inputs=['x', 'y'], outputs=['z'], direction="LEFT"
        )

        x = np.array([16, 4, 1]).astype(np.uint32)
        y = np.array([1, 2, 3]).astype(np.uint32)
        z = x << y  # expected output [32, 16, 8]
        expect(node, inputs=[x, y], outputs=[z], name='test_bitshift_left_uint32')

    @staticmethod
    def export_left_unit64():  # type: () -> None
        node = onnx.helper.make_node(
            'BitShift', inputs=['x', 'y'], outputs=['z'], direction="LEFT"
        )

        x = np.array([16, 4, 1]).astype(np.uint64)
        y = np.array([1, 2, 3]).astype(np.uint64)
        z = x << y  # expected output [32, 16, 8]
        expect(node, inputs=[x, y], outputs=[z], name='test_bitshift_left_uint64')


if __name__ == '__main__':
    BitShift.export_left_unit16()
    BitShift.export_left_unit32()
    BitShift.export_left_unit64()
    BitShift.export_left_unit8()
    BitShift.export_right_unit16()
    BitShift.export_right_unit32()
    BitShift.export_right_unit64()
    BitShift.export_right_unit8()
