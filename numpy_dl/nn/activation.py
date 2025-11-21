"""Activation function layers."""

from numpy_dl.core.module import Module
from numpy_dl.core.tensor import Tensor
from numpy_dl.core import functional as F


class ReLU(Module):
    """Rectified Linear Unit activation function."""

    def forward(self, x: Tensor) -> Tensor:
        """Apply ReLU activation."""
        return F.relu(x)

    def __repr__(self):
        return "ReLU()"


class LeakyReLU(Module):
    """Leaky Rectified Linear Unit activation function."""

    def __init__(self, negative_slope: float = 0.01):
        """
        Initialize LeakyReLU.

        Args:
            negative_slope: Slope for negative values
        """
        super().__init__()
        self.negative_slope = negative_slope

    def forward(self, x: Tensor) -> Tensor:
        """Apply Leaky ReLU activation."""
        return F.leaky_relu(x, self.negative_slope)

    def __repr__(self):
        return f"LeakyReLU(negative_slope={self.negative_slope})"


class Sigmoid(Module):
    """Sigmoid activation function."""

    def forward(self, x: Tensor) -> Tensor:
        """Apply sigmoid activation."""
        return F.sigmoid(x)

    def __repr__(self):
        return "Sigmoid()"


class Tanh(Module):
    """Hyperbolic tangent activation function."""

    def forward(self, x: Tensor) -> Tensor:
        """Apply tanh activation."""
        return F.tanh(x)

    def __repr__(self):
        return "Tanh()"


class Softmax(Module):
    """Softmax activation function."""

    def __init__(self, axis: int = -1):
        """
        Initialize Softmax.

        Args:
            axis: Dimension along which to apply softmax
        """
        super().__init__()
        self.axis = axis

    def forward(self, x: Tensor) -> Tensor:
        """Apply softmax activation."""
        return F.softmax(x, self.axis)

    def __repr__(self):
        return f"Softmax(axis={self.axis})"


class LogSoftmax(Module):
    """Log-Softmax activation function."""

    def __init__(self, axis: int = -1):
        """
        Initialize LogSoftmax.

        Args:
            axis: Dimension along which to apply log-softmax
        """
        super().__init__()
        self.axis = axis

    def forward(self, x: Tensor) -> Tensor:
        """Apply log-softmax activation."""
        return F.log_softmax(x, self.axis)

    def __repr__(self):
        return f"LogSoftmax(axis={self.axis})"
