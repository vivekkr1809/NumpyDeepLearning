"""Dropout layer for regularization."""

from numpy_dl.core.module import Module
from numpy_dl.core.tensor import Tensor
from numpy_dl.core import functional as F


class Dropout(Module):
    """
    Dropout layer for regularization.

    During training, randomly zeroes some elements of the input tensor
    with probability p using samples from a Bernoulli distribution.
    """

    def __init__(self, p: float = 0.5):
        """
        Initialize Dropout layer.

        Args:
            p: Probability of an element being zeroed
        """
        super().__init__()
        if p < 0 or p > 1:
            raise ValueError(f"Dropout probability must be between 0 and 1, got {p}")
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor

        Returns:
            Output tensor with dropout applied
        """
        return F.dropout(x, self.p, self.training)

    def __repr__(self):
        return f"Dropout(p={self.p})"
