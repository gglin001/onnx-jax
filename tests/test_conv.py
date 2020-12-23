import jax.numpy as jnp
import numpy as np  # type: ignore
import onnx
from onnx import defs, helper
from onnx.helper import make_opsetid
from onnx_jax.backend import run_node

from tests.tools import _cosin_sim, _gen_random


class Conv:

    @staticmethod
    def export():  # type: () -> None

        x = jnp.array([[[[0., 1., 2., 3., 4.],  # (1, 1, 5, 5) input tensor
                        [5., 6., 7., 8., 9.],
                        [10., 11., 12., 13., 14.],
                        [15., 16., 17., 18., 19.],
                        [20., 21., 22., 23., 24.]]]]).astype(jnp.float32)
        W = jnp.array([[[[1., 1., 1.],  # (1, 1, 3, 3) tensor for convolution weights
                        [1., 1., 1.],
                        [1., 1., 1.]]]]).astype(jnp.float32)

        # Convolution with padding
        node_with_padding = onnx.helper.make_node(
            'Conv',
            inputs=['x', 'W'],
            outputs=['y'],
            kernel_shape=[3, 3],
            # Default values for other attributes: strides=[1, 1], dilations=[1, 1], groups=1
            pads=[1, 1, 1, 1],
        )
        y_with_padding = jnp.array([[[[12., 21., 27., 33., 24.],  # (1, 1, 5, 5) output tensor
                                     [33., 54., 63., 72., 51.],
                                     [63., 99., 108., 117., 81.],
                                     [93., 144., 153., 162., 111.],
                                     [72., 111., 117., 123., 84.]]]]).astype(jnp.float32)
        expect(node_with_padding, inputs=[x, W], outputs=[y_with_padding],
               name='test_basic_conv_with_padding')

        # Convolution without padding
        node_without_padding = onnx.helper.make_node(
            'Conv',
            inputs=['x', 'W'],
            outputs=['y'],
            kernel_shape=[3, 3],
            # Default values for other attributes: strides=[1, 1], dilations=[1, 1], groups=1
            pads=[0, 0, 0, 0],
        )
        y_without_padding = jnp.array([[[[54., 63., 72.],  # (1, 1, 3, 3) output tensor
                                        [99., 108., 117.],
                                        [144., 153., 162.]]]]).astype(jnp.float32)
        expect(node_without_padding, inputs=[x, W], outputs=[y_without_padding],
               name='test_basic_conv_without_padding')

    @staticmethod
    def export_conv_with_strides():  # type: () -> None

        x = jnp.array([[[[0., 1., 2., 3., 4.],  # (1, 1, 7, 5) input tensor
                        [5., 6., 7., 8., 9.],
                        [10., 11., 12., 13., 14.],
                        [15., 16., 17., 18., 19.],
                        [20., 21., 22., 23., 24.],
                        [25., 26., 27., 28., 29.],
                        [30., 31., 32., 33., 34.]]]]).astype(jnp.float32)
        W = jnp.array([[[[1., 1., 1.],  # (1, 1, 3, 3) tensor for convolution weights
                        [1., 1., 1.],
                        [1., 1., 1.]]]]).astype(jnp.float32)

        # Convolution with strides=2 and padding
        node_with_padding = onnx.helper.make_node(
            'Conv',
            inputs=['x', 'W'],
            outputs=['y'],
            kernel_shape=[3, 3],
            pads=[1, 1, 1, 1],
            strides=[2, 2],  # Default values for other attributes: dilations=[1, 1], groups=1
        )
        y_with_padding = jnp.array([[[[12., 27., 24.],  # (1, 1, 4, 3) output tensor
                                     [63., 108., 81.],
                                     [123., 198., 141.],
                                     [112., 177., 124.]]]]).astype(jnp.float32)
        expect(node_with_padding, inputs=[x, W], outputs=[y_with_padding],
               name='test_conv_with_strides_padding')

        # Convolution with strides=2 and no padding
        node_without_padding = onnx.helper.make_node(
            'Conv',
            inputs=['x', 'W'],
            outputs=['y'],
            kernel_shape=[3, 3],
            pads=[0, 0, 0, 0],
            strides=[2, 2],  # Default values for other attributes: dilations=[1, 1], groups=1
        )
        y_without_padding = jnp.array([[[[54., 72.],  # (1, 1, 3, 2) output tensor
                                        [144., 162.],
                                        [234., 252.]]]]).astype(jnp.float32)
        expect(node_without_padding, inputs=[x, W], outputs=[y_without_padding],
               name='test_conv_with_strides_no_padding')

        # Convolution with strides=2 and padding only along one dimension (the H dimension in NxCxHxW tensor)
        node_with_asymmetric_padding = onnx.helper.make_node(
            'Conv',
            inputs=['x', 'W'],
            outputs=['y'],
            kernel_shape=[3, 3],
            pads=[1, 0, 1, 0],
            strides=[2, 2],  # Default values for other attributes: dilations=[1, 1], groups=1
        )
        y_with_asymmetric_padding = jnp.array([[[[21., 33.],  # (1, 1, 4, 2) output tensor
                                                [99., 117.],
                                                [189., 207.],
                                                [171., 183.]]]]).astype(jnp.float32)
        expect(node_with_asymmetric_padding, inputs=[x, W], outputs=[y_with_asymmetric_padding],
               name='test_conv_with_strides_and_asymmetric_padding')


def expect(node, inputs, outputs, **kwargs):
    outputs_jax = run_node(node, inputs)

    print(f"golden size: {outputs[0].shape}, output shape: {outputs_jax[0].shape}")
    if list(outputs[0].shape) == list(outputs_jax[0].shape):
        sim = _cosin_sim(outputs_jax[0], jnp.asarray(outputs[0]))
        print(sim)
    else:
        print('failed')


Conv.export()
Conv.export_conv_with_strides()

