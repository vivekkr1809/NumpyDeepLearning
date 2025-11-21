"""Multi-Layer Perceptron (MLP) model."""

from typing import List
from numpy_dl.core.module import Module, Sequential
from numpy_dl.nn import Linear, ReLU, Dropout
from numpy_dl.core.tensor import Tensor


class MLP(Module):
    """
    Multi-Layer Perceptron (MLP).

    A fully connected feedforward neural network with configurable layers.
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int],
        output_size: int,
        dropout: float = 0.0,
        activation: str = 'relu',
    ):
        """
        Initialize MLP.

        Args:
            input_size: Size of input features
            hidden_sizes: List of hidden layer sizes
            output_size: Size of output
            dropout: Dropout probability
            activation: Activation function ('relu', 'tanh', 'sigmoid')
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size

        # Build layers
        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.append(Linear(prev_size, hidden_size))

            if activation == 'relu':
                layers.append(ReLU())
            elif activation == 'tanh':
                from numpy_dl.nn import Tanh
                layers.append(Tanh())
            elif activation == 'sigmoid':
                from numpy_dl.nn import Sigmoid
                layers.append(Sigmoid())

            if dropout > 0:
                layers.append(Dropout(dropout))

            prev_size = hidden_size

        # Output layer
        layers.append(Linear(prev_size, output_size))

        self.layers = Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, input_size)

        Returns:
            Output tensor of shape (batch_size, output_size)
        """
        return self.layers(x)

    def __repr__(self):
        return (
            f"MLP(input_size={self.input_size}, hidden_sizes={self.hidden_sizes}, "
            f"output_size={self.output_size})"
        )
