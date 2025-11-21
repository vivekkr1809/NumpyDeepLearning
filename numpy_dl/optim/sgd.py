"""Stochastic Gradient Descent optimizer."""

import numpy as np
from typing import Iterator
from numpy_dl.core.parameter import Parameter
from numpy_dl.optim.optimizer import Optimizer
from numpy_dl.utils.device import get_array_module


class SGD(Optimizer):
    """
    Stochastic Gradient Descent optimizer.

    Implements SGD with optional momentum and weight decay.
    """

    def __init__(
        self,
        params: Iterator[Parameter],
        lr: float = 0.01,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        dampening: float = 0.0,
        nesterov: bool = False,
    ):
        """
        Initialize SGD optimizer.

        Args:
            params: Iterator of parameters to optimize
            lr: Learning rate
            momentum: Momentum factor
            weight_decay: Weight decay (L2 penalty)
            dampening: Dampening for momentum
            nesterov: Whether to use Nesterov momentum
        """
        super().__init__(params, lr)
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.dampening = dampening
        self.nesterov = nesterov

        # Initialize velocity buffers for momentum
        self.velocity = [None] * len(self.params)

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

            # Apply momentum
            if self.momentum != 0:
                if self.velocity[i] is None:
                    self.velocity[i] = xp.zeros_like(param.data)

                v = self.velocity[i]
                v = self.momentum * v + (1 - self.dampening) * grad
                self.velocity[i] = v

                if self.nesterov:
                    grad = grad + self.momentum * v
                else:
                    grad = v

            # Update parameters
            param.data -= self.lr * grad

    def __repr__(self):
        return (
            f"SGD(lr={self.lr}, momentum={self.momentum}, "
            f"weight_decay={self.weight_decay}, nesterov={self.nesterov})"
        )
