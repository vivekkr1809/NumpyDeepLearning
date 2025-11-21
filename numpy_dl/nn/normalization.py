"""Normalization layers."""

import numpy as np
from numpy_dl.core.module import Module
from numpy_dl.core.parameter import Parameter
from numpy_dl.core.tensor import Tensor, zeros, ones
from numpy_dl.core import functional as F
from numpy_dl.utils.device import get_array_module


class BatchNorm1d(Module):
    """
    Batch Normalization for 1D/2D inputs.

    Applies Batch Normalization over a 2D or 3D input.
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
    ):
        """
        Initialize BatchNorm1d layer.

        Args:
            num_features: Number of features/channels
            eps: Small constant for numerical stability
            momentum: Momentum for running mean/var
            affine: Whether to learn affine parameters
        """
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine

        if affine:
            self.gamma = Parameter(np.ones(num_features, dtype=np.float32))
            self.beta = Parameter(np.zeros(num_features, dtype=np.float32))
        else:
            self.gamma = Tensor(np.ones(num_features, dtype=np.float32))
            self.beta = Tensor(np.zeros(num_features, dtype=np.float32))

        # Running statistics (not trainable)
        self.running_mean = Tensor(np.zeros(num_features, dtype=np.float32))
        self.running_var = Tensor(np.ones(num_features, dtype=np.float32))

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, num_features) or (batch_size, num_features, length)

        Returns:
            Normalized tensor
        """
        return F.batch_norm(
            x,
            self.gamma,
            self.beta,
            self.running_mean,
            self.running_var,
            self.training,
            self.momentum,
            self.eps,
        )

    def __repr__(self):
        return (
            f"BatchNorm1d(num_features={self.num_features}, eps={self.eps}, "
            f"momentum={self.momentum}, affine={self.affine})"
        )


class BatchNorm2d(Module):
    """
    Batch Normalization for 4D inputs (images).

    Applies Batch Normalization over a 4D input (N, C, H, W).
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
    ):
        """
        Initialize BatchNorm2d layer.

        Args:
            num_features: Number of channels
            eps: Small constant for numerical stability
            momentum: Momentum for running mean/var
            affine: Whether to learn affine parameters
        """
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine

        if affine:
            self.gamma = Parameter(np.ones(num_features, dtype=np.float32))
            self.beta = Parameter(np.zeros(num_features, dtype=np.float32))
        else:
            self.gamma = Tensor(np.ones(num_features, dtype=np.float32))
            self.beta = Tensor(np.zeros(num_features, dtype=np.float32))

        # Running statistics (not trainable)
        self.running_mean = Tensor(np.zeros(num_features, dtype=np.float32))
        self.running_var = Tensor(np.ones(num_features, dtype=np.float32))

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, num_features, height, width)

        Returns:
            Normalized tensor
        """
        # Reshape to (N, C, H*W) for batch norm
        batch_size, channels, height, width = x.shape
        x_reshaped = x.reshape(batch_size, channels, height * width)

        out = F.batch_norm(
            x_reshaped,
            self.gamma,
            self.beta,
            self.running_mean,
            self.running_var,
            self.training,
            self.momentum,
            self.eps,
        )

        return out.reshape(batch_size, channels, height, width)

    def __repr__(self):
        return (
            f"BatchNorm2d(num_features={self.num_features}, eps={self.eps}, "
            f"momentum={self.momentum}, affine={self.affine})"
        )


class LayerNorm(Module):
    """
    Layer Normalization.

    Applies Layer Normalization over a mini-batch of inputs.
    """

    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
    ):
        """
        Initialize LayerNorm layer.

        Args:
            normalized_shape: Input shape from an expected input
            eps: Small constant for numerical stability
            elementwise_affine: Whether to learn affine parameters
        """
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if elementwise_affine:
            self.gamma = Parameter(np.ones(normalized_shape, dtype=np.float32))
            self.beta = Parameter(np.zeros(normalized_shape, dtype=np.float32))
        else:
            self.gamma = None
            self.beta = None

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor

        Returns:
            Normalized tensor
        """
        xp = get_array_module(x.data)

        # Compute mean and variance along last dimension
        mean = xp.mean(x.data, axis=-1, keepdims=True)
        var = xp.var(x.data, axis=-1, keepdims=True)

        # Normalize
        x_norm = (x.data - mean) / xp.sqrt(var + self.eps)

        if self.elementwise_affine:
            out_data = self.gamma.data * x_norm + self.beta.data
        else:
            out_data = x_norm

        out = Tensor(
            out_data,
            requires_grad=x.requires_grad,
            device=x.device,
            _children=(x,) if not self.elementwise_affine else (x, self.gamma, self.beta),
            _op='layer_norm'
        )

        def _backward():
            N = self.normalized_shape
            dx_norm = out.grad

            if self.elementwise_affine:
                if self.gamma.requires_grad:
                    grad_gamma = xp.sum(dx_norm * x_norm, axis=tuple(range(x.ndim - 1)))
                    self.gamma.grad = grad_gamma if self.gamma.grad is None else self.gamma.grad + grad_gamma

                if self.beta.requires_grad:
                    grad_beta = xp.sum(dx_norm, axis=tuple(range(x.ndim - 1)))
                    self.beta.grad = grad_beta if self.beta.grad is None else self.beta.grad + grad_beta

                dx_norm = dx_norm * self.gamma.data

            if x.requires_grad:
                dvar = xp.sum(dx_norm * (x.data - mean) * -0.5 * (var + self.eps) ** (-1.5), axis=-1, keepdims=True)
                dmean = xp.sum(dx_norm * -1 / xp.sqrt(var + self.eps), axis=-1, keepdims=True) + dvar * xp.sum(-2 * (x.data - mean), axis=-1, keepdims=True) / N

                grad_x = dx_norm / xp.sqrt(var + self.eps) + dvar * 2 * (x.data - mean) / N + dmean / N
                x.grad = grad_x if x.grad is None else x.grad + grad_x

        out._backward = _backward
        return out

    def __repr__(self):
        return (
            f"LayerNorm(normalized_shape={self.normalized_shape}, eps={self.eps}, "
            f"elementwise_affine={self.elementwise_affine})"
        )
