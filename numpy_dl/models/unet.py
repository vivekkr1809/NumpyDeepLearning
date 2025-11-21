"""U-Net architecture for image segmentation."""

from numpy_dl.core.module import Module, Sequential
from numpy_dl.nn import Conv2d, MaxPool2d, ReLU, BatchNorm2d
from numpy_dl.core.tensor import Tensor
from numpy_dl.utils.device import get_array_module


class DoubleConv(Module):
    """
    Double convolution block: Conv -> BN -> ReLU -> Conv -> BN -> ReLU.
    """

    def __init__(self, in_channels: int, out_channels: int):
        """
        Initialize DoubleConv.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
        """
        super().__init__()
        self.conv = Sequential(
            Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            BatchNorm2d(out_channels),
            ReLU(),
            Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            BatchNorm2d(out_channels),
            ReLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        return self.conv(x)


class Down(Module):
    """
    Downsampling block: MaxPool -> DoubleConv.
    """

    def __init__(self, in_channels: int, out_channels: int):
        """
        Initialize Down block.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
        """
        super().__init__()
        self.down = Sequential(
            MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        return self.down(x)


class Up(Module):
    """
    Upsampling block: Upsample -> Conv -> Concat -> DoubleConv.
    """

    def __init__(self, in_channels: int, out_channels: int):
        """
        Initialize Up block.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
        """
        super().__init__()
        self.up = Conv2d(in_channels, in_channels // 2, kernel_size=2, stride=1)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        """
        Forward pass with skip connection.

        Args:
            x1: Input from previous layer (to be upsampled)
            x2: Skip connection from encoder

        Returns:
            Upsampled and concatenated features
        """
        xp = get_array_module(x1.data)

        # Upsample x1 using nearest neighbor (simplified)
        # In practice, would use proper upsampling
        batch, channels, h, w = x1.shape
        upsampled = xp.repeat(xp.repeat(x1.data, 2, axis=2), 2, axis=3)
        x1 = Tensor(upsampled, requires_grad=x1.requires_grad, device=x1.device)

        # Crop x2 if needed to match x1 size
        diff_h = x2.shape[2] - x1.shape[2]
        diff_w = x2.shape[3] - x1.shape[3]

        if diff_h > 0 or diff_w > 0:
            x2_data = x2.data[
                :,
                :,
                diff_h // 2: x2.shape[2] - (diff_h - diff_h // 2),
                diff_w // 2: x2.shape[3] - (diff_w - diff_w // 2),
            ]
            x2 = Tensor(x2_data, requires_grad=x2.requires_grad, device=x2.device)

        # Concatenate along channel dimension
        concat_data = xp.concatenate([x1.data, x2.data], axis=1)
        x = Tensor(concat_data, requires_grad=x1.requires_grad or x2.requires_grad, device=x1.device)

        return self.conv(x)


class UNet(Module):
    """
    U-Net architecture for image segmentation.

    A U-shaped encoder-decoder network with skip connections.
    """

    def __init__(
        self,
        input_channels: int = 3,
        num_classes: int = 2,
        features: list = [64, 128, 256, 512],
    ):
        """
        Initialize U-Net.

        Args:
            input_channels: Number of input channels
            num_classes: Number of output classes
            features: List of feature channels for each level
        """
        super().__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes

        # Encoder (downsampling path)
        self.inc = DoubleConv(input_channels, features[0])

        self.down1 = Down(features[0], features[1])
        self.down2 = Down(features[1], features[2])
        self.down3 = Down(features[2], features[3])

        # Bottleneck
        self.down4 = Down(features[3], features[3] * 2)

        # Decoder (upsampling path)
        self.up1 = Up(features[3] * 2, features[3])
        self.up2 = Up(features[3], features[2])
        self.up3 = Up(features[2], features[1])
        self.up4 = Up(features[1], features[0])

        # Output convolution
        self.outc = Conv2d(features[0], num_classes, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width)

        Returns:
            Output tensor of shape (batch_size, num_classes, height, width)
        """
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Decoder with skip connections
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        # Output
        x = self.outc(x)

        return x

    def __repr__(self):
        return f"UNet(input_channels={self.input_channels}, num_classes={self.num_classes})"
