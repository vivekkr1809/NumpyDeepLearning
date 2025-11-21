"""Convolutional Neural Network (CNN) models."""

from typing import List, Tuple
from numpy_dl.core.module import Module, Sequential
from numpy_dl.nn import Conv2d, MaxPool2d, Linear, ReLU, Dropout, BatchNorm2d
from numpy_dl.core.tensor import Tensor


class SimpleCNN(Module):
    """
    Simple Convolutional Neural Network.

    A basic CNN with configurable convolutional layers followed by fully connected layers.
    """

    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        conv_channels: List[int] = [32, 64, 128],
        fc_sizes: List[int] = [512],
        input_size: Tuple[int, int] = (28, 28),
        dropout: float = 0.5,
    ):
        """
        Initialize SimpleCNN.

        Args:
            input_channels: Number of input channels (e.g., 1 for grayscale, 3 for RGB)
            num_classes: Number of output classes
            conv_channels: List of output channels for each conv layer
            fc_sizes: List of fully connected layer sizes
            input_size: Input image size (H, W)
            dropout: Dropout probability
        """
        super().__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes

        # Build convolutional layers
        conv_layers = []
        prev_channels = input_channels

        for out_channels in conv_channels:
            conv_layers.extend([
                Conv2d(prev_channels, out_channels, kernel_size=3, padding=1),
                ReLU(),
                MaxPool2d(kernel_size=2, stride=2),
            ])
            prev_channels = out_channels

        self.conv_layers = Sequential(*conv_layers)

        # Calculate size after conv layers
        h, w = input_size
        for _ in conv_channels:
            h, w = h // 2, w // 2
        flatten_size = prev_channels * h * w

        # Build fully connected layers
        fc_layers = []
        prev_size = flatten_size

        for fc_size in fc_sizes:
            fc_layers.extend([
                Linear(prev_size, fc_size),
                ReLU(),
                Dropout(dropout),
            ])
            prev_size = fc_size

        fc_layers.append(Linear(prev_size, num_classes))
        self.fc_layers = Sequential(*fc_layers)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width)

        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        x = self.conv_layers(x)
        x = x.reshape(x.shape[0], -1)  # Flatten
        x = self.fc_layers(x)
        return x

    def __repr__(self):
        return f"SimpleCNN(input_channels={self.input_channels}, num_classes={self.num_classes})"


class VGGBlock(Module):
    """VGG-style convolutional block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_convs: int,
        use_batch_norm: bool = False,
    ):
        """
        Initialize VGG block.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            num_convs: Number of conv layers in the block
            use_batch_norm: Whether to use batch normalization
        """
        super().__init__()
        layers = []

        for i in range(num_convs):
            in_ch = in_channels if i == 0 else out_channels
            layers.append(Conv2d(in_ch, out_channels, kernel_size=3, padding=1))

            if use_batch_norm:
                layers.append(BatchNorm2d(out_channels))

            layers.append(ReLU())

        layers.append(MaxPool2d(kernel_size=2, stride=2))
        self.block = Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        return self.block(x)


class VGG(Module):
    """
    VGG-style network.

    A deeper CNN architecture inspired by VGG.
    """

    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        architecture: List[Tuple[int, int]] = [(64, 2), (128, 2), (256, 3), (512, 3)],
        use_batch_norm: bool = False,
        dropout: float = 0.5,
    ):
        """
        Initialize VGG.

        Args:
            input_channels: Number of input channels
            num_classes: Number of output classes
            architecture: List of (channels, num_convs) for each block
            use_batch_norm: Whether to use batch normalization
            dropout: Dropout probability
        """
        super().__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes

        # Build VGG blocks
        blocks = []
        prev_channels = input_channels

        for out_channels, num_convs in architecture:
            blocks.append(VGGBlock(prev_channels, out_channels, num_convs, use_batch_norm))
            prev_channels = out_channels

        self.features = Sequential(*blocks)

        # Classifier
        self.classifier = Sequential(
            Linear(prev_channels * 7 * 7, 4096),  # Assuming 224x224 input -> 7x7 after pooling
            ReLU(),
            Dropout(dropout),
            Linear(4096, 4096),
            ReLU(),
            Dropout(dropout),
            Linear(4096, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width)

        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        x = self.features(x)
        x = x.reshape(x.shape[0], -1)
        x = self.classifier(x)
        return x

    def __repr__(self):
        return f"VGG(input_channels={self.input_channels}, num_classes={self.num_classes})"
