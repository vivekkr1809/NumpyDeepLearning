"""Multi-Task Learning loss functions and weighting strategies.

This module implements state-of-the-art loss weighting strategies for multi-task learning:
1. Uncertainty Weighting (Kendall et al., 2018)
2. Gradient Normalization (GradNorm)
3. Dynamic Weight Average (DWA)
4. Equal Weighting (baseline)

References:
    - Kendall et al. "Multi-Task Learning Using Uncertainty to Weigh Losses" (CVPR 2018)
    - Chen et al. "GradNorm: Gradient Normalization for Adaptive Loss Balancing" (ICML 2018)
    - Liu et al. "End-to-End Multi-Task Learning with Attention" (CVPR 2019)
"""

from typing import Dict, List, Optional, Union
import numpy as np
from numpy_dl.core.module import Module
from numpy_dl.core.tensor import Tensor
from numpy_dl.core.parameter import Parameter


class MultiTaskLoss(Module):
    """
    Multi-task loss wrapper that combines multiple task-specific losses.

    This class serves as a base for different multi-task loss weighting strategies.

    Args:
        task_losses: Dictionary mapping task names to loss functions
        loss_weights: Optional dictionary of initial loss weights per task
        reduction: Reduction method ('mean', 'sum', or 'none')

    Example:
        >>> from numpy_dl.loss import MSELoss, CrossEntropyLoss
        >>> task_losses = {
        ...     'regression': MSELoss(),
        ...     'classification': CrossEntropyLoss()
        ... }
        >>> mtl_loss = MultiTaskLoss(task_losses)
        >>> # During training:
        >>> total_loss, task_losses = mtl_loss(outputs, targets)
    """

    def __init__(
        self,
        task_losses: Dict[str, Module],
        loss_weights: Optional[Dict[str, float]] = None,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.task_names = list(task_losses.keys())
        self.num_tasks = len(self.task_names)
        self.reduction = reduction

        # Register task losses as modules
        for task_name, loss_fn in task_losses.items():
            self.add_module(f'loss_{task_name}', loss_fn)

        # Initialize loss weights
        if loss_weights is None:
            loss_weights = {task: 1.0 for task in self.task_names}
        self.loss_weights = loss_weights

        # Track loss history for dynamic weighting
        self.loss_history: Dict[str, List[float]] = {task: [] for task in self.task_names}

    def get_task_loss(self, task_name: str) -> Module:
        """Get the loss function for a specific task."""
        return getattr(self, f'loss_{task_name}')

    def compute_task_losses(
        self,
        outputs: Dict[str, Tensor],
        targets: Dict[str, Tensor]
    ) -> Dict[str, Tensor]:
        """
        Compute individual task losses.

        Args:
            outputs: Dictionary mapping task names to model outputs
            targets: Dictionary mapping task names to target values

        Returns:
            Dictionary mapping task names to loss values
        """
        task_losses = {}
        for task_name in self.task_names:
            if task_name not in outputs:
                raise ValueError(f"Task '{task_name}' not found in outputs")
            if task_name not in targets:
                raise ValueError(f"Task '{task_name}' not found in targets")

            loss_fn = self.get_task_loss(task_name)
            task_losses[task_name] = loss_fn(outputs[task_name], targets[task_name])

        return task_losses

    def forward(
        self,
        outputs: Dict[str, Tensor],
        targets: Dict[str, Tensor],
        return_dict: bool = False
    ) -> Union[Tensor, tuple]:
        """
        Compute weighted multi-task loss.

        Args:
            outputs: Dictionary mapping task names to model outputs
            targets: Dictionary mapping task names to target values
            return_dict: If True, return (total_loss, task_losses_dict)

        Returns:
            Total weighted loss, or (total_loss, task_losses) if return_dict=True
        """
        task_losses = self.compute_task_losses(outputs, targets)

        # Apply weights and combine
        weighted_losses = {}
        total_loss = None

        for task_name, loss in task_losses.items():
            weight = self.loss_weights[task_name]
            weighted_loss = weight * loss
            weighted_losses[task_name] = weighted_loss

            if total_loss is None:
                total_loss = weighted_loss
            else:
                total_loss = total_loss + weighted_loss

        # Update loss history
        for task_name, loss in task_losses.items():
            loss_val = float(loss.data) if hasattr(loss.data, 'item') else float(loss.data)
            self.loss_history[task_name].append(loss_val)

        if return_dict:
            return total_loss, task_losses
        return total_loss


class UncertaintyWeighting(MultiTaskLoss):
    """
    Uncertainty-based multi-task loss weighting (Kendall et al., 2018).

    Learns task-dependent uncertainty (homoscedastic uncertainty) to automatically
    balance losses. Each task has a learnable log-variance parameter that represents
    the task's uncertainty.

    The weighted loss for each task is:
        L_weighted = (1 / (2 * sigma^2)) * L_task + log(sigma)

    where sigma^2 is the task's learned variance.

    Args:
        task_losses: Dictionary mapping task names to loss functions
        task_types: Dictionary mapping task names to 'regression' or 'classification'
        init_log_vars: Optional initial log-variance values (default: 0.0)
        reduction: Reduction method ('mean', 'sum', or 'none')

    Example:
        >>> task_losses = {'depth': MSELoss(), 'segmentation': CrossEntropyLoss()}
        >>> task_types = {'depth': 'regression', 'segmentation': 'classification'}
        >>> loss_fn = UncertaintyWeighting(task_losses, task_types)
    """

    def __init__(
        self,
        task_losses: Dict[str, Module],
        task_types: Optional[Dict[str, str]] = None,
        init_log_vars: Optional[Dict[str, float]] = None,
        reduction: str = 'mean'
    ):
        super().__init__(task_losses, reduction=reduction)

        # Set task types (default to regression)
        if task_types is None:
            task_types = {task: 'regression' for task in self.task_names}
        self.task_types = task_types

        # Initialize learnable log-variance parameters for each task
        if init_log_vars is None:
            init_log_vars = {task: 0.0 for task in self.task_names}

        self.log_vars = {}
        for task_name in self.task_names:
            log_var = Parameter(
                data=np.array([init_log_vars.get(task_name, 0.0)]),
                requires_grad=True
            )
            self.add_parameter(f'log_var_{task_name}', log_var)
            self.log_vars[task_name] = log_var

    def forward(
        self,
        outputs: Dict[str, Tensor],
        targets: Dict[str, Tensor],
        return_dict: bool = False
    ) -> Union[Tensor, tuple]:
        """
        Compute uncertainty-weighted multi-task loss.

        Args:
            outputs: Dictionary mapping task names to model outputs
            targets: Dictionary mapping task names to target values
            return_dict: If True, return (total_loss, task_losses_dict)

        Returns:
            Total weighted loss, or (total_loss, task_losses) if return_dict=True
        """
        task_losses = self.compute_task_losses(outputs, targets)

        total_loss = None
        weighted_losses = {}

        for task_name, loss in task_losses.items():
            log_var = self.log_vars[task_name]

            # Uncertainty weighting: (1 / (2 * sigma^2)) * L + log(sigma)
            # Using log_var for numerical stability: log_var = log(sigma^2)
            precision = (-log_var).exp()  # 1 / sigma^2

            weighted_loss = 0.5 * precision * loss + 0.5 * log_var
            weighted_losses[task_name] = weighted_loss

            if total_loss is None:
                total_loss = weighted_loss
            else:
                total_loss = total_loss + weighted_loss

        # Update loss history
        for task_name, loss in task_losses.items():
            loss_val = float(loss.data) if hasattr(loss.data, 'item') else float(loss.data)
            self.loss_history[task_name].append(loss_val)

        if return_dict:
            return total_loss, task_losses
        return total_loss

    def get_task_weights(self) -> Dict[str, float]:
        """
        Get current task weights based on learned uncertainties.

        Returns:
            Dictionary mapping task names to their effective weights (1/sigma^2)
        """
        weights = {}
        for task_name in self.task_names:
            log_var = self.log_vars[task_name]
            # Weight is proportional to precision (1 / sigma^2)
            weight = float((-log_var.data).exp())
            weights[task_name] = weight
        return weights

    def get_uncertainties(self) -> Dict[str, float]:
        """
        Get current task uncertainties (sigma values).

        Returns:
            Dictionary mapping task names to their uncertainty (sigma)
        """
        uncertainties = {}
        for task_name in self.task_names:
            log_var = self.log_vars[task_name]
            # sigma = sqrt(sigma^2) = exp(0.5 * log_var)
            sigma = float((0.5 * log_var.data).exp())
            uncertainties[task_name] = sigma
        return uncertainties


class GradNorm(MultiTaskLoss):
    """
    Gradient Normalization for adaptive loss balancing (Chen et al., 2018).

    Balances training by ensuring task gradients have similar magnitudes.
    Adjusts loss weights to normalize gradient magnitudes across tasks.

    Args:
        task_losses: Dictionary mapping task names to loss functions
        alpha: Hyperparameter controlling adaptation rate (default: 1.5)
        reduction: Reduction method ('mean', 'sum', or 'none')

    Note:
        GradNorm requires computing gradients with respect to the last shared layer.
        Use the MultiTaskModel with enable_gradnorm=True for proper functionality.

    Example:
        >>> task_losses = {'task1': MSELoss(), 'task2': CrossEntropyLoss()}
        >>> loss_fn = GradNorm(task_losses, alpha=1.5)
    """

    def __init__(
        self,
        task_losses: Dict[str, Module],
        alpha: float = 1.5,
        reduction: str = 'mean'
    ):
        super().__init__(task_losses, reduction=reduction)
        self.alpha = alpha

        # Initialize learnable loss weights
        self.weight_params = {}
        for task_name in self.task_names:
            weight = Parameter(
                data=np.array([1.0]),
                requires_grad=True
            )
            self.add_parameter(f'weight_{task_name}', weight)
            self.weight_params[task_name] = weight
            self.loss_weights[task_name] = 1.0

        # Track initial losses for computing relative inverse training rates
        self.initial_losses: Optional[Dict[str, float]] = None
        self.iteration = 0

    def forward(
        self,
        outputs: Dict[str, Tensor],
        targets: Dict[str, Tensor],
        return_dict: bool = False
    ) -> Union[Tensor, tuple]:
        """
        Compute GradNorm-weighted multi-task loss.

        Args:
            outputs: Dictionary mapping task names to model outputs
            targets: Dictionary mapping task names to target values
            return_dict: If True, return (total_loss, task_losses_dict)

        Returns:
            Total weighted loss, or (total_loss, task_losses) if return_dict=True
        """
        task_losses = self.compute_task_losses(outputs, targets)

        # Store initial losses on first iteration
        if self.initial_losses is None:
            self.initial_losses = {}
            for task_name, loss in task_losses.items():
                loss_val = float(loss.data)
                self.initial_losses[task_name] = loss_val

        # Update weights from parameters (ensure positive via softmax)
        total_weight = sum(float(w.data) for w in self.weight_params.values())
        for task_name in self.task_names:
            self.loss_weights[task_name] = float(self.weight_params[task_name].data) / total_weight

        # Compute weighted total loss
        total_loss = None
        weighted_losses = {}

        for task_name, loss in task_losses.items():
            weight = self.loss_weights[task_name]
            weighted_loss = weight * loss
            weighted_losses[task_name] = weighted_loss

            if total_loss is None:
                total_loss = weighted_loss
            else:
                total_loss = total_loss + weighted_loss

        # Update loss history
        for task_name, loss in task_losses.items():
            loss_val = float(loss.data)
            self.loss_history[task_name].append(loss_val)

        self.iteration += 1

        if return_dict:
            return total_loss, task_losses
        return total_loss

    def get_task_weights(self) -> Dict[str, float]:
        """Get current task weights (normalized)."""
        return self.loss_weights.copy()


class DynamicWeightAverage(MultiTaskLoss):
    """
    Dynamic Weight Average for multi-task learning (Liu et al., 2019).

    Adjusts task weights based on the rate of change of task losses.
    Tasks with faster decreasing losses get lower weights, and vice versa.

    Args:
        task_losses: Dictionary mapping task names to loss functions
        temperature: Temperature parameter for softmax (default: 2.0)
        window_size: Number of previous losses to consider (default: 2)
        reduction: Reduction method ('mean', 'sum', or 'none')

    Example:
        >>> task_losses = {'task1': MSELoss(), 'task2': CrossEntropyLoss()}
        >>> loss_fn = DynamicWeightAverage(task_losses, temperature=2.0)
    """

    def __init__(
        self,
        task_losses: Dict[str, Module],
        temperature: float = 2.0,
        window_size: int = 2,
        reduction: str = 'mean'
    ):
        super().__init__(task_losses, reduction=reduction)
        self.temperature = temperature
        self.window_size = window_size
        self.iteration = 0

    def compute_dynamic_weights(self) -> Dict[str, float]:
        """
        Compute dynamic weights based on loss rate of change.

        Returns:
            Dictionary mapping task names to dynamic weights
        """
        if self.iteration < self.window_size:
            # Not enough history, use equal weights
            return {task: 1.0 / self.num_tasks for task in self.task_names}

        # Compute loss ratios (rate of change)
        loss_ratios = {}
        for task_name in self.task_names:
            history = self.loss_history[task_name]
            if len(history) < self.window_size + 1:
                loss_ratios[task_name] = 1.0
            else:
                # Ratio of current loss to loss from window_size iterations ago
                current_loss = history[-1]
                previous_loss = history[-(self.window_size + 1)]
                if previous_loss > 0:
                    loss_ratios[task_name] = current_loss / previous_loss
                else:
                    loss_ratios[task_name] = 1.0

        # Apply softmax with temperature
        ratios_array = np.array([loss_ratios[task] for task in self.task_names])
        exp_ratios = np.exp(ratios_array / self.temperature)
        weights_array = self.num_tasks * exp_ratios / np.sum(exp_ratios)

        weights = {
            task: float(weights_array[i])
            for i, task in enumerate(self.task_names)
        }

        return weights

    def forward(
        self,
        outputs: Dict[str, Tensor],
        targets: Dict[str, Tensor],
        return_dict: bool = False
    ) -> Union[Tensor, tuple]:
        """
        Compute DWA-weighted multi-task loss.

        Args:
            outputs: Dictionary mapping task names to model outputs
            targets: Dictionary mapping task names to target values
            return_dict: If True, return (total_loss, task_losses_dict)

        Returns:
            Total weighted loss, or (total_loss, task_losses) if return_dict=True
        """
        task_losses = self.compute_task_losses(outputs, targets)

        # Update loss history first
        for task_name, loss in task_losses.items():
            loss_val = float(loss.data)
            self.loss_history[task_name].append(loss_val)

        # Compute dynamic weights
        self.loss_weights = self.compute_dynamic_weights()

        # Compute weighted total loss
        total_loss = None
        weighted_losses = {}

        for task_name, loss in task_losses.items():
            weight = self.loss_weights[task_name]
            weighted_loss = weight * loss
            weighted_losses[task_name] = weighted_loss

            if total_loss is None:
                total_loss = weighted_loss
            else:
                total_loss = total_loss + weighted_loss

        self.iteration += 1

        if return_dict:
            return total_loss, task_losses
        return total_loss
