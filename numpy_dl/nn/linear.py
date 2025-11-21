"""Linear (fully connected) layer."""

import numpy as np
from numpy_dl.core.module import Module
from numpy_dl.core.parameter import Parameter
from numpy_dl.core.tensor import Tensor
from typing import Optional


class Linear(Module):
    """
    Linear (fully connected) layer.

    Applies a linear transformation: y = xW^T + b
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        """
        Initialize Linear layer.

        Args:
            in_features: Size of input features
            out_features: Size of output features
            bias: Whether to include bias term
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Initialize weights with He initialization
        limit = np.sqrt(2.0 / in_features)
        self.weight = Parameter(
            np.random.randn(out_features, in_features).astype(np.float32) * limit
        )

        if bias:
            self.bias = Parameter(np.zeros(out_features, dtype=np.float32))
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, in_features) or (*, in_features)

        Returns:
            Output tensor of shape (batch_size, out_features) or (*, out_features)
        """
        # x @ W.T
        out = x @ self.weight.T

        if self.bias is not None:
            out = out + self.bias

        return out

    def __repr__(self):
        return f"Linear(in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None})"
