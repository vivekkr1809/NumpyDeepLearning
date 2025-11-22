"""Base optimizer class."""

from typing import List, Iterator
import numpy as np
from numpy_dl.core.parameter import Parameter
from numpy_dl.utils.logging import get_logger


class Optimizer:
    """
    Base class for all optimizers.

    All optimizers should subclass this class and implement the step() method.
    """

    def __init__(self, params: Iterator[Parameter], lr: float):
        """
        Initialize optimizer.

        Args:
            params: Iterator of parameters to optimize
            lr: Learning rate
        """
        self.params = list(params)
        self.lr = lr
        self.logger = get_logger('optimizer')
        self._step_count = 0

        self.logger.info(
            "Initialized optimizer",
            optimizer_type=self.__class__.__name__,
            learning_rate=lr,
            num_parameters=len(self.params)
        )

    def zero_grad(self):
        """Zero out all parameter gradients."""
        for param in self.params:
            param.zero_grad()

    def step(self):
        """
        Perform a single optimization step.

        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement step()")

    def _check_gradients(self):
        """
        Check for gradient issues (NaN, Inf, vanishing, exploding).

        Logs warnings if gradient issues are detected.
        """
        has_nan = False
        has_inf = False
        max_grad = 0.0
        min_grad = float('inf')
        none_grad_count = 0

        for i, param in enumerate(self.params):
            if param.grad is None:
                none_grad_count += 1
                continue

            grad_data = param.grad.data if hasattr(param.grad, 'data') else param.grad

            # Check for NaN/Inf
            if np.any(np.isnan(grad_data)):
                has_nan = True
                self.logger.error(
                    "NaN gradient detected",
                    parameter_index=i,
                    param_shape=param.shape,
                    step=self._step_count
                )

            if np.any(np.isinf(grad_data)):
                has_inf = True
                self.logger.error(
                    "Inf gradient detected",
                    parameter_index=i,
                    param_shape=param.shape,
                    step=self._step_count
                )

            # Track gradient magnitude
            grad_norm = np.linalg.norm(grad_data)
            max_grad = max(max_grad, grad_norm)
            min_grad = min(min_grad, grad_norm)

        # Log gradient statistics
        if self._step_count % 100 == 0:  # Log every 100 steps
            self.logger.debug(
                "Gradient statistics",
                step=self._step_count,
                max_grad_norm=float(max_grad),
                min_grad_norm=float(min_grad) if min_grad != float('inf') else 0.0,
                none_grad_count=none_grad_count
            )

        # Warn about vanishing gradients
        if min_grad < 1e-7 and min_grad != float('inf'):
            self.logger.warning(
                "Vanishing gradients detected",
                min_grad_norm=float(min_grad),
                step=self._step_count
            )

        # Warn about exploding gradients
        if max_grad > 1000.0:
            self.logger.warning(
                "Exploding gradients detected",
                max_grad_norm=float(max_grad),
                step=self._step_count
            )

        if has_nan or has_inf:
            self.logger.error(
                "Invalid gradients detected",
                has_nan=has_nan,
                has_inf=has_inf,
                step=self._step_count
            )

    def __repr__(self):
        return f"{self.__class__.__name__}(lr={self.lr})"
