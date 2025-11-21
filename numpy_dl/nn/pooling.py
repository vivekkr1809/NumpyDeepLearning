"""Pooling layers."""

from numpy_dl.core.module import Module
from numpy_dl.core.tensor import Tensor
from numpy_dl.core import functional as F
from typing import Union, Tuple, Optional


class MaxPool2d(Module):
    """
    2D max pooling layer.

    Applies a 2D max pooling over an input signal.
    """

    def __init__(
        self,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Optional[Union[int, Tuple[int, int]]] = None,
    ):
        """
        Initialize MaxPool2d layer.

        Args:
            kernel_size: Size of the pooling window
            stride: Stride of the pooling window (defaults to kernel_size)
        """
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.kernel_size = kernel_size

        if stride is None:
            stride = kernel_size
        elif isinstance(stride, int):
            stride = (stride, stride)
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width)

        Returns:
            Output tensor of shape (batch_size, channels, out_height, out_width)
        """
        return F.max_pool2d(x, self.kernel_size, self.stride)

    def __repr__(self):
        return f"MaxPool2d(kernel_size={self.kernel_size}, stride={self.stride})"


class AvgPool2d(Module):
    """
    2D average pooling layer.

    Applies a 2D average pooling over an input signal.
    """

    def __init__(
        self,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Optional[Union[int, Tuple[int, int]]] = None,
    ):
        """
        Initialize AvgPool2d layer.

        Args:
            kernel_size: Size of the pooling window
            stride: Stride of the pooling window (defaults to kernel_size)
        """
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.kernel_size = kernel_size

        if stride is None:
            stride = kernel_size
        elif isinstance(stride, int):
            stride = (stride, stride)
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width)

        Returns:
            Output tensor of shape (batch_size, channels, out_height, out_width)
        """
        return F.avg_pool2d(x, self.kernel_size, self.stride)

    def __repr__(self):
        return f"AvgPool2d(kernel_size={self.kernel_size}, stride={self.stride})"


class AdaptiveAvgPool2d(Module):
    """
    2D adaptive average pooling layer.

    Applies a 2D adaptive average pooling over an input signal.
    The output is of specified size regardless of input size.
    """

    def __init__(self, output_size: Union[int, Tuple[int, int]]):
        """
        Initialize AdaptiveAvgPool2d layer.

        Args:
            output_size: Target output size (H, W) or H for square output
        """
        super().__init__()
        if isinstance(output_size, int):
            output_size = (output_size, output_size)
        self.output_size = output_size

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width)

        Returns:
            Output tensor of shape (batch_size, channels, out_height, out_width)
        """
        batch_size, channels, in_h, in_w = x.shape
        out_h, out_w = self.output_size

        stride_h = in_h // out_h
        stride_w = in_w // out_w
        kernel_h = in_h - (out_h - 1) * stride_h
        kernel_w = in_w - (out_w - 1) * stride_w

        return F.avg_pool2d(x, (kernel_h, kernel_w), (stride_h, stride_w))

    def __repr__(self):
        return f"AdaptiveAvgPool2d(output_size={self.output_size})"
