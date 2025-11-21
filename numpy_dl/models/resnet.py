"""ResNet (Residual Network) architecture."""

from numpy_dl.core.module import Module, Sequential
from numpy_dl.nn import Conv2d, BatchNorm2d, ReLU, MaxPool2d, AdaptiveAvgPool2d, Linear
from numpy_dl.core.tensor import Tensor


class BasicBlock(Module):
    """
    Basic ResNet block.

    Two 3x3 convolutions with a skip connection.
    """

    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, downsample: Module = None):
        """
        Initialize BasicBlock.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            stride: Stride for first convolution
            downsample: Downsampling layer for skip connection
        """
        super().__init__()

        self.conv1 = Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = BatchNorm2d(out_channels)
        self.relu = ReLU()
        self.conv2 = Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass with residual connection."""
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)

        return out


class Bottleneck(Module):
    """
    Bottleneck ResNet block.

    1x1 -> 3x3 -> 1x1 convolutions with a skip connection.
    """

    expansion = 4

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, downsample: Module = None):
        """
        Initialize Bottleneck.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels (before expansion)
            stride: Stride for 3x3 convolution
            downsample: Downsampling layer for skip connection
        """
        super().__init__()

        self.conv1 = Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(out_channels)
        self.conv2 = Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = BatchNorm2d(out_channels)
        self.conv3 = Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = BatchNorm2d(out_channels * self.expansion)
        self.relu = ReLU()
        self.downsample = downsample

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass with residual connection."""
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)

        return out


class ResNet(Module):
    """
    ResNet architecture.

    Residual Neural Network with configurable depth.
    """

    def __init__(
        self,
        block_type: str,
        layers: list,
        num_classes: int = 1000,
        input_channels: int = 3,
    ):
        """
        Initialize ResNet.

        Args:
            block_type: Type of block ('basic' or 'bottleneck')
            layers: List of number of blocks in each layer
            num_classes: Number of output classes
            input_channels: Number of input channels
        """
        super().__init__()

        if block_type == 'basic':
            self.block = BasicBlock
        elif block_type == 'bottleneck':
            self.block = Bottleneck
        else:
            raise ValueError(f"Invalid block type: {block_type}")

        self.in_channels = 64

        # Initial convolution
        self.conv1 = Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = BatchNorm2d(64)
        self.relu = ReLU()
        self.maxpool = MaxPool2d(kernel_size=3, stride=2)

        # Residual layers
        self.layer1 = self._make_layer(64, layers[0], stride=1)
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)

        # Classification head
        self.avgpool = AdaptiveAvgPool2d((1, 1))
        self.fc = Linear(512 * self.block.expansion, num_classes)

    def _make_layer(self, out_channels: int, num_blocks: int, stride: int = 1) -> Sequential:
        """
        Create a residual layer.

        Args:
            out_channels: Number of output channels
            num_blocks: Number of blocks in this layer
            stride: Stride for first block

        Returns:
            Sequential module containing the blocks
        """
        downsample = None
        if stride != 1 or self.in_channels != out_channels * self.block.expansion:
            downsample = Sequential(
                Conv2d(self.in_channels, out_channels * self.block.expansion, kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(out_channels * self.block.expansion),
            )

        layers = []
        layers.append(self.block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * self.block.expansion

        for _ in range(1, num_blocks):
            layers.append(self.block(self.in_channels, out_channels))

        return Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width)

        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x


def resnet18(num_classes: int = 1000, input_channels: int = 3) -> ResNet:
    """Create ResNet-18 model."""
    return ResNet('basic', [2, 2, 2, 2], num_classes, input_channels)


def resnet34(num_classes: int = 1000, input_channels: int = 3) -> ResNet:
    """Create ResNet-34 model."""
    return ResNet('basic', [3, 4, 6, 3], num_classes, input_channels)


def resnet50(num_classes: int = 1000, input_channels: int = 3) -> ResNet:
    """Create ResNet-50 model."""
    return ResNet('bottleneck', [3, 4, 6, 3], num_classes, input_channels)


def resnet101(num_classes: int = 1000, input_channels: int = 3) -> ResNet:
    """Create ResNet-101 model."""
    return ResNet('bottleneck', [3, 4, 23, 3], num_classes, input_channels)


def resnet152(num_classes: int = 1000, input_channels: int = 3) -> ResNet:
    """Create ResNet-152 model."""
    return ResNet('bottleneck', [3, 8, 36, 3], num_classes, input_channels)
