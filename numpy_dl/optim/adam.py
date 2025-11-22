"""Adam optimizer."""

import numpy as np
from typing import Iterator
from numpy_dl.core.parameter import Parameter
from numpy_dl.optim.optimizer import Optimizer
from numpy_dl.utils.device import get_array_module


class Adam(Optimizer):
    """
    Adam optimizer.

    Implements Adam algorithm (Adaptive Moment Estimation).
    """

    def __init__(
        self,
        params: Iterator[Parameter],
        lr: float = 0.001,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        """
        Initialize Adam optimizer.

        Args:
            params: Iterator of parameters to optimize
            lr: Learning rate
            betas: Coefficients for computing running averages (beta1, beta2)
            eps: Term added for numerical stability
            weight_decay: Weight decay (L2 penalty)
        """
        super().__init__(params, lr)
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay

        # Initialize moment estimates
        self.m = [None] * len(self.params)  # First moment
        self.v = [None] * len(self.params)  # Second moment
        self.t = 0  # Time step

    def step(self):
        """Perform a single optimization step."""
        self._step_count += 1
        self.t += 1

        # Check gradients for numerical issues
        try:
            self._check_gradients()
        except Exception as e:
            self.logger.warning("Gradient check failed", error=str(e))

        for i, param in enumerate(self.params):
            if param.grad is None:
                continue

            try:
                xp = get_array_module(param.data)
                grad = param.grad

                # Apply weight decay
                if self.weight_decay != 0:
                    grad = grad + self.weight_decay * param.data

                # Initialize moments if needed
                if self.m[i] is None:
                    self.m[i] = xp.zeros_like(param.data)
                    self.v[i] = xp.zeros_like(param.data)

                # Update biased first moment estimate
                self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad

                # Update biased second raw moment estimate
                self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad ** 2)

                # Compute bias-corrected first moment estimate
                m_hat = self.m[i] / (1 - self.beta1 ** self.t)

                # Compute bias-corrected second raw moment estimate
                v_hat = self.v[i] / (1 - self.beta2 ** self.t)

                # Update parameters
                param.data -= self.lr * m_hat / (xp.sqrt(v_hat) + self.eps)

            except Exception as e:
                self.logger.exception(
                    "Error updating parameter",
                    parameter_index=i,
                    param_shape=param.shape,
                    step=self._step_count,
                    error=str(e)
                )
                raise

    def __repr__(self):
        return (
            f"Adam(lr={self.lr}, betas=({self.beta1}, {self.beta2}), "
            f"eps={self.eps}, weight_decay={self.weight_decay})"
        )


class AdamW(Optimizer):
    """
    AdamW optimizer.

    Implements AdamW algorithm (Adam with decoupled weight decay).
    """

    def __init__(
        self,
        params: Iterator[Parameter],
        lr: float = 0.001,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        """
        Initialize AdamW optimizer.

        Args:
            params: Iterator of parameters to optimize
            lr: Learning rate
            betas: Coefficients for computing running averages (beta1, beta2)
            eps: Term added for numerical stability
            weight_decay: Weight decay coefficient
        """
        super().__init__(params, lr)
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay

        # Initialize moment estimates
        self.m = [None] * len(self.params)  # First moment
        self.v = [None] * len(self.params)  # Second moment
        self.t = 0  # Time step

    def step(self):
        """Perform a single optimization step."""
        self._step_count += 1
        self.t += 1

        # Check gradients for numerical issues
        try:
            self._check_gradients()
        except Exception as e:
            self.logger.warning("Gradient check failed", error=str(e))

        for i, param in enumerate(self.params):
            if param.grad is None:
                continue

            try:
                xp = get_array_module(param.data)
                grad = param.grad

                # Initialize moments if needed
                if self.m[i] is None:
                    self.m[i] = xp.zeros_like(param.data)
                    self.v[i] = xp.zeros_like(param.data)

                # Update biased first moment estimate
                self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad

                # Update biased second raw moment estimate
                self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad ** 2)

                # Compute bias-corrected first moment estimate
                m_hat = self.m[i] / (1 - self.beta1 ** self.t)

                # Compute bias-corrected second raw moment estimate
                v_hat = self.v[i] / (1 - self.beta2 ** self.t)

                # Update parameters with decoupled weight decay
                param.data -= self.lr * (m_hat / (xp.sqrt(v_hat) + self.eps) + self.weight_decay * param.data)

            except Exception as e:
                self.logger.exception(
                    "Error updating parameter",
                    parameter_index=i,
                    param_shape=param.shape,
                    step=self._step_count,
                    error=str(e)
                )
                raise

    def __repr__(self):
        return (
            f"AdamW(lr={self.lr}, betas=({self.beta1}, {self.beta2}), "
            f"eps={self.eps}, weight_decay={self.weight_decay})"
        )
