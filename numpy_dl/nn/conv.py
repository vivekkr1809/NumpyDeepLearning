"""Convolutional layers."""

import numpy as np
from numpy_dl.core.module import Module
from numpy_dl.core.parameter import Parameter
from numpy_dl.core.tensor import Tensor
from numpy_dl.core import functional as F
from typing import Union, Tuple, Optional


class Conv2d(Module):
    """
    2D convolutional layer.

    Applies a 2D convolution over an input signal composed of several input planes.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        bias: bool = True,
    ):
        """
        Initialize Conv2d layer.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Size of the convolving kernel
            stride: Stride of the convolution
            padding: Zero-padding added to both sides of the input
            bias: Whether to add a learnable bias
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.kernel_size = kernel_size

        if isinstance(stride, int):
            stride = (stride, stride)
        self.stride = stride

        if isinstance(padding, int):
            padding = (padding, padding)
        self.padding = padding

        # Initialize weights with He initialization
        fan_in = in_channels * kernel_size[0] * kernel_size[1]
        limit = np.sqrt(2.0 / fan_in)
        self.weight = Parameter(
            np.random.randn(out_channels, in_channels, kernel_size[0], kernel_size[1]).astype(np.float32) * limit
        )

        if bias:
            self.bias = Parameter(np.zeros(out_channels, dtype=np.float32))
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, in_channels, height, width)

        Returns:
            Output tensor of shape (batch_size, out_channels, out_height, out_width)
        """
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding)

    def __repr__(self):
        return (
            f"Conv2d(in_channels={self.in_channels}, out_channels={self.out_channels}, "
            f"kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}, "
            f"bias={self.bias is not None})"
        )
