"""RMSprop optimizer."""

import numpy as np
from typing import Iterator
from numpy_dl.core.parameter import Parameter
from numpy_dl.optim.optimizer import Optimizer
from numpy_dl.utils.device import get_array_module


class RMSprop(Optimizer):
    """
    RMSprop optimizer.

    Implements RMSprop algorithm (Root Mean Square Propagation).
    """

    def __init__(
        self,
        params: Iterator[Parameter],
        lr: float = 0.01,
        alpha: float = 0.99,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        momentum: float = 0.0,
    ):
        """
        Initialize RMSprop optimizer.

        Args:
            params: Iterator of parameters to optimize
            lr: Learning rate
            alpha: Smoothing constant
            eps: Term added for numerical stability
            weight_decay: Weight decay (L2 penalty)
            momentum: Momentum factor
        """
        super().__init__(params, lr)
        self.alpha = alpha
        self.eps = eps
        self.weight_decay = weight_decay
        self.momentum = momentum

        # Initialize squared gradient running average
        self.square_avg = [None] * len(self.params)

        # Initialize momentum buffer
        if momentum > 0:
            self.momentum_buffer = [None] * len(self.params)
        else:
            self.momentum_buffer = None

    def step(self):
        """Perform a single optimization step."""
        for i, param in enumerate(self.params):
            if param.grad is None:
                continue

            xp = get_array_module(param.data)
            grad = param.grad

            # Apply weight decay
            if self.weight_decay != 0:
                grad = grad + self.weight_decay * param.data

            # Initialize squared average if needed
            if self.square_avg[i] is None:
                self.square_avg[i] = xp.zeros_like(param.data)

            # Update squared gradient running average
            self.square_avg[i] = self.alpha * self.square_avg[i] + (1 - self.alpha) * (grad ** 2)

            # Compute adaptive learning rate
            avg = xp.sqrt(self.square_avg[i]) + self.eps

            # Apply momentum if specified
            if self.momentum > 0:
                if self.momentum_buffer[i] is None:
                    self.momentum_buffer[i] = xp.zeros_like(param.data)

                buf = self.momentum_buffer[i]
                buf = self.momentum * buf + grad / avg
                self.momentum_buffer[i] = buf
                param.data -= self.lr * buf
            else:
                param.data -= self.lr * grad / avg

    def __repr__(self):
        return (
            f"RMSprop(lr={self.lr}, alpha={self.alpha}, eps={self.eps}, "
            f"weight_decay={self.weight_decay}, momentum={self.momentum})"
        )
