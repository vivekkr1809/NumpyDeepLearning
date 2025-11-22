"""Embedding layer for neural networks."""

import numpy as np
from numpy_dl.core.module import Module
from numpy_dl.core.parameter import Parameter
from numpy_dl.core.tensor import Tensor


class Embedding(Module):
    """
    Embedding layer that maps discrete indices to continuous vectors.

    Creates a lookup table of embeddings of a fixed dictionary and size.

    Args:
        num_embeddings: Size of the dictionary (vocabulary size)
        embedding_dim: Dimension of each embedding vector
        padding_idx: If specified, entries at padding_idx do not contribute to gradient

    Attributes:
        weight: Embedding matrix of shape (num_embeddings, embedding_dim)

    Example:
        >>> embedding = Embedding(num_embeddings=1000, embedding_dim=128)
        >>> indices = Tensor(np.array([[1, 2, 3], [4, 5, 6]]))
        >>> output = embedding(indices)  # (2, 3, 128)
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int = None
    ):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx

        # Initialize embedding weights
        # Using normal distribution with std=1/sqrt(embedding_dim)
        std = 1.0 / np.sqrt(embedding_dim)
        self.weight = Parameter(
            np.random.randn(num_embeddings, embedding_dim).astype(np.float32) * std
        )

        # Set padding embedding to zero if specified
        if padding_idx is not None:
            self.weight.data[padding_idx] = 0

    def forward(self, indices: Tensor) -> Tensor:
        """
        Forward pass.

        Args:
            indices: Tensor of indices (*, ) where * is any shape
                    Values must be in range [0, num_embeddings)

        Returns:
            Embedded vectors (*, embedding_dim)
        """
        # Get the indices as numpy array
        idx = indices.data if hasattr(indices, 'data') else indices

        # Convert to integer indices
        idx = idx.astype(np.int64)

        # Lookup embeddings
        output_data = self.weight.data[idx]

        # Create output tensor
        output = Tensor(
            output_data,
            requires_grad=self.weight.requires_grad,
            device=indices.device if hasattr(indices, 'device') else None
        )

        # Setup backward pass
        if self.weight.requires_grad:
            def _backward():
                # Accumulate gradients for each embedding
                if self.weight.grad is None:
                    self.weight.grad = np.zeros_like(self.weight.data)

                # Add gradients at each index
                np.add.at(self.weight.grad, idx.flatten(), output.grad.reshape(-1, self.embedding_dim))

                # Zero out gradient for padding index
                if self.padding_idx is not None:
                    self.weight.grad[self.padding_idx] = 0

            output._backward = _backward
            output._prev = {self.weight}

        return output

    def __repr__(self):
        return (
            f"Embedding(num_embeddings={self.num_embeddings}, "
            f"embedding_dim={self.embedding_dim}"
            f"{f', padding_idx={self.padding_idx}' if self.padding_idx is not None else ''})"
        )
