"""Multi-Task Learning utilities and training helpers.

This module provides utilities for training and evaluating multi-task learning models,
including training loops, metric tracking, and visualization tools.
"""

from typing import Dict, List, Optional, Tuple, Callable, Any
import numpy as np
from numpy_dl.core.tensor import Tensor
from numpy_dl.core.module import Module
from numpy_dl.data import DataLoader
from numpy_dl.utils.logging import get_logger, ContextLogger


class MultiTaskMetrics:
    """
    Track and compute metrics for multi-task learning.

    Maintains separate metrics for each task and computes aggregate statistics.

    Args:
        task_names: List of task names
        metric_fns: Dictionary mapping task names to metric functions
            Each metric function should take (predictions, targets) and return a float

    Example:
        >>> from numpy_dl.utils.metrics import accuracy
        >>> metrics = MultiTaskMetrics(
        ...     task_names=['digit', 'parity'],
        ...     metric_fns={'digit': accuracy, 'parity': accuracy}
        ... )
        >>> metrics.update({'digit': pred1, 'parity': pred2}, {'digit': tgt1, 'parity': tgt2})
        >>> results = metrics.compute()
    """

    def __init__(
        self,
        task_names: List[str],
        metric_fns: Dict[str, Callable[[np.ndarray, np.ndarray], float]]
    ):
        self.task_names = task_names
        self.metric_fns = metric_fns
        self.reset()

    def reset(self):
        """Reset all metrics."""
        self.predictions: Dict[str, List[np.ndarray]] = {
            task: [] for task in self.task_names
        }
        self.targets: Dict[str, List[np.ndarray]] = {
            task: [] for task in self.task_names
        }
        self.losses: Dict[str, List[float]] = {
            task: [] for task in self.task_names
        }

    def update(
        self,
        predictions: Dict[str, Tensor],
        targets: Dict[str, Tensor],
        losses: Optional[Dict[str, Tensor]] = None
    ):
        """
        Update metrics with new batch of predictions and targets.

        Args:
            predictions: Dictionary mapping task names to prediction tensors
            targets: Dictionary mapping task names to target tensors
            losses: Optional dictionary of loss values per task
        """
        for task in self.task_names:
            if task in predictions and task in targets:
                pred = predictions[task].numpy() if isinstance(predictions[task], Tensor) else predictions[task]
                tgt = targets[task].numpy() if isinstance(targets[task], Tensor) else targets[task]

                self.predictions[task].append(pred)
                self.targets[task].append(tgt)

                if losses is not None and task in losses:
                    loss_val = losses[task].numpy() if isinstance(losses[task], Tensor) else losses[task]
                    self.losses[task].append(float(loss_val))

    def compute(self) -> Dict[str, Dict[str, float]]:
        """
        Compute final metrics for all tasks.

        Returns:
            Dictionary with structure:
            {
                'task1': {'metric': value, 'loss': value},
                'task2': {'metric': value, 'loss': value},
                'average': {'metric': avg_value, 'loss': avg_value}
            }
        """
        results = {}

        for task in self.task_names:
            if len(self.predictions[task]) == 0:
                continue

            # Concatenate all predictions and targets
            all_preds = np.concatenate(self.predictions[task], axis=0)
            all_targets = np.concatenate(self.targets[task], axis=0)

            # Compute metric
            metric_fn = self.metric_fns.get(task)
            if metric_fn is not None:
                metric_value = metric_fn(all_preds, all_targets)
            else:
                metric_value = 0.0

            # Compute average loss
            avg_loss = np.mean(self.losses[task]) if len(self.losses[task]) > 0 else 0.0

            results[task] = {
                'metric': metric_value,
                'loss': avg_loss
            }

        # Compute averages across tasks
        if len(results) > 0:
            avg_metric = np.mean([r['metric'] for r in results.values()])
            avg_loss = np.mean([r['loss'] for r in results.values()])
            results['average'] = {
                'metric': avg_metric,
                'loss': avg_loss
            }

        return results

    def compute_task_metric(self, task: str) -> Optional[float]:
        """
        Compute metric for a specific task.

        Args:
            task: Task name

        Returns:
            Metric value or None if no data
        """
        if len(self.predictions[task]) == 0:
            return None

        all_preds = np.concatenate(self.predictions[task], axis=0)
        all_targets = np.concatenate(self.targets[task], axis=0)

        metric_fn = self.metric_fns.get(task)
        if metric_fn is not None:
            return metric_fn(all_preds, all_targets)
        return None


class MultiTaskTrainer:
    """
    Trainer for multi-task learning models.

    Provides a complete training loop with support for:
    - Multiple loss weighting strategies
    - Task-specific metrics
    - Learning rate scheduling
    - Early stopping
    - Checkpointing

    Args:
        model: Multi-task model
        loss_fn: Multi-task loss function
        optimizer: Optimizer
        device: Device to train on ('cpu' or 'cuda')
        metric_fns: Optional dictionary of metric functions per task
        grad_clip: Optional gradient clipping value

    Example:
        >>> from numpy_dl.optim import Adam
        >>> from numpy_dl.loss.multitask import UncertaintyWeighting
        >>> trainer = MultiTaskTrainer(
        ...     model=model,
        ...     loss_fn=UncertaintyWeighting(task_losses, task_types),
        ...     optimizer=Adam(model.parameters(), lr=0.001),
        ...     device='cuda'
        ... )
        >>> history = trainer.fit(train_loader, val_loader, epochs=10)
    """

    def __init__(
        self,
        model: Module,
        loss_fn: Module,
        optimizer: Any,
        device: str = 'cpu',
        metric_fns: Optional[Dict[str, Callable]] = None,
        grad_clip: Optional[float] = None
    ):
        self.model = model.to(device)
        self.loss_fn = loss_fn.to(device)
        self.optimizer = optimizer
        self.device = device
        self.metric_fns = metric_fns or {}
        self.grad_clip = grad_clip
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': []
        }
        self.logger = get_logger('training')
        self.logger.info(
            "Initialized MultiTaskTrainer",
            device=device,
            num_tasks=len(metric_fns),
            grad_clip=grad_clip,
            optimizer=type(optimizer).__name__
        )

    def _parse_batch(self, batch):
        """
        Parse batch into inputs and targets.

        Handles multiple batch formats:
        - Tuple/list: (inputs, targets) or (inputs, task1_target, task2_target, ...)
        - Dict: {'input'/'inputs': ..., 'task1': ..., 'task2': ...}

        Args:
            batch: Batch data in various formats

        Returns:
            Tuple of (inputs, targets_dict)

        Raises:
            ValueError: If batch format is not supported
        """
        try:
            if isinstance(batch, (list, tuple)):
                if len(batch) == 2:
                    inputs, targets = batch
                else:
                    inputs = batch[0]
                    targets = {task: batch[i + 1] for i, task in enumerate(self.metric_fns.keys())}
            elif isinstance(batch, dict):
                inputs = batch.get('input', batch.get('inputs'))
                if inputs is None:
                    self.logger.error(
                        "Batch dict missing 'input' or 'inputs' key",
                        batch_keys=list(batch.keys())
                    )
                    raise ValueError("Batch dict must contain 'input' or 'inputs' key")
                targets = {k: v for k, v in batch.items() if k not in ['input', 'inputs']}
            else:
                self.logger.error(
                    "Unsupported batch format",
                    batch_type=type(batch).__name__,
                    expected_types="tuple, list, or dict"
                )
                raise ValueError(f"Unsupported batch format: {type(batch)}")

            return inputs, targets

        except Exception as e:
            self.logger.exception(
                "Failed to parse batch",
                batch_type=type(batch).__name__,
                error=str(e)
            )
            raise

    def train_epoch(
        self,
        train_loader: DataLoader,
        verbose: bool = True
    ) -> Tuple[float, Dict[str, float]]:
        """
        Train for one epoch.

        Args:
            train_loader: Training data loader
            verbose: Whether to print progress

        Returns:
            (average_loss, task_metrics)
        """
        self.logger.debug("Starting training epoch", num_batches=len(train_loader))

        self.model.train()
        epoch_loss = 0.0
        num_batches = 0
        gradient_clipped_count = 0

        # Initialize metrics tracker
        if self.metric_fns:
            metrics_tracker = MultiTaskMetrics(
                task_names=list(self.metric_fns.keys()),
                metric_fns=self.metric_fns
            )

        try:
            for batch_idx, batch in enumerate(train_loader):
                try:
                    # Parse batch into inputs and targets
                    inputs, targets = self._parse_batch(batch)

                    # Move to device
                    if isinstance(inputs, Tensor):
                        inputs = inputs.to(self.device)
                    if isinstance(targets, dict):
                        targets = {k: v.to(self.device) if isinstance(v, Tensor) else v
                                  for k, v in targets.items()}

                    # Forward pass
                    self.optimizer.zero_grad()
                    outputs = self.model(inputs)

                    # Compute loss
                    loss, task_losses = self.loss_fn(outputs, targets, return_dict=True)

                    # Check for NaN/Inf in loss
                    loss_val = float(loss.data)
                    if np.isnan(loss_val) or np.isinf(loss_val):
                        self.logger.error(
                            "Invalid loss value detected",
                            batch_idx=batch_idx,
                            loss_value=loss_val,
                            task_losses={k: float(v.data) for k, v in task_losses.items()}
                        )
                        raise ValueError(f"Invalid loss value: {loss_val}")

                    # Backward pass
                    loss.backward()

                    # Gradient clipping with logging
                    if self.grad_clip is not None:
                        max_grad_norm = 0.0
                        for param in self.model.parameters():
                            if param.grad is not None:
                                grad_norm = np.linalg.norm(param.grad.data)
                                max_grad_norm = max(max_grad_norm, grad_norm)
                                if grad_norm > self.grad_clip:
                                    param.grad.data = param.grad.data * (self.grad_clip / grad_norm)
                                    gradient_clipped_count += 1

                        if max_grad_norm > self.grad_clip:
                            self.logger.debug(
                                "Gradient clipping applied",
                                batch_idx=batch_idx,
                                max_grad_norm=float(max_grad_norm),
                                clip_threshold=self.grad_clip
                            )

                    self.optimizer.step()

                    # Track metrics
                    epoch_loss += loss_val
                    num_batches += 1

                    if self.metric_fns:
                        metrics_tracker.update(outputs, targets, task_losses)

                    if verbose and (batch_idx + 1) % 10 == 0:
                        print(f"  Batch {batch_idx + 1}/{len(train_loader)}, Loss: {loss_val:.4f}")

                except Exception as e:
                    self.logger.exception(
                        "Error during batch training",
                        batch_idx=batch_idx,
                        error=str(e)
                    )
                    raise

            avg_loss = epoch_loss / num_batches
            metrics = metrics_tracker.compute() if self.metric_fns else {}

            self.logger.info(
                "Training epoch completed",
                avg_loss=avg_loss,
                num_batches=num_batches,
                gradient_clipped_batches=gradient_clipped_count
            )

            return avg_loss, metrics

        except Exception as e:
            self.logger.exception(
                "Training epoch failed",
                error=str(e),
                batches_completed=num_batches
            )
            raise

    def validate(
        self,
        val_loader: DataLoader
    ) -> Tuple[float, Dict[str, float]]:
        """
        Validate the model.

        Args:
            val_loader: Validation data loader

        Returns:
            (average_loss, task_metrics)
        """
        self.logger.debug("Starting validation", num_batches=len(val_loader))

        self.model.eval()
        epoch_loss = 0.0
        num_batches = 0

        # Initialize metrics tracker
        if self.metric_fns:
            metrics_tracker = MultiTaskMetrics(
                task_names=list(self.metric_fns.keys()),
                metric_fns=self.metric_fns
            )

        try:
            for batch_idx, batch in enumerate(val_loader):
                try:
                    # Parse batch into inputs and targets
                    inputs, targets = self._parse_batch(batch)

                    # Move to device
                    if isinstance(inputs, Tensor):
                        inputs = inputs.to(self.device)
                    if isinstance(targets, dict):
                        targets = {k: v.to(self.device) if isinstance(v, Tensor) else v
                                  for k, v in targets.items()}

                    # Forward pass (no gradients needed)
                    outputs = self.model(inputs)
                    loss, task_losses = self.loss_fn(outputs, targets, return_dict=True)

                    # Check for NaN/Inf in validation loss
                    loss_val = float(loss.data)
                    if np.isnan(loss_val) or np.isinf(loss_val):
                        self.logger.warning(
                            "Invalid loss value during validation",
                            batch_idx=batch_idx,
                            loss_value=loss_val,
                            task_losses={k: float(v.data) for k, v in task_losses.items()}
                        )

                    epoch_loss += loss_val
                    num_batches += 1

                    if self.metric_fns:
                        metrics_tracker.update(outputs, targets, task_losses)

                except Exception as e:
                    self.logger.exception(
                        "Error during batch validation",
                        batch_idx=batch_idx,
                        error=str(e)
                    )
                    raise

            avg_loss = epoch_loss / num_batches
            metrics = metrics_tracker.compute() if self.metric_fns else {}

            self.logger.info(
                "Validation completed",
                avg_loss=avg_loss,
                num_batches=num_batches
            )

            return avg_loss, metrics

        except Exception as e:
            self.logger.exception(
                "Validation failed",
                error=str(e),
                batches_completed=num_batches
            )
            raise

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 10,
        verbose: bool = True
    ) -> Dict[str, List]:
        """
        Train the model for multiple epochs.

        Args:
            train_loader: Training data loader
            val_loader: Optional validation data loader
            epochs: Number of epochs to train
            verbose: Whether to print progress

        Returns:
            Training history dictionary
        """
        self.logger.info(
            "Starting training",
            epochs=epochs,
            train_batches=len(train_loader),
            val_batches=len(val_loader) if val_loader else 0
        )

        try:
            with ContextLogger(self.logger, "training", epochs=epochs):
                for epoch in range(epochs):
                    if verbose:
                        print(f"\nEpoch {epoch + 1}/{epochs}")
                        print("-" * 50)

                    self.logger.info(f"Starting epoch {epoch + 1}/{epochs}")

                    # Train
                    train_loss, train_metrics = self.train_epoch(train_loader, verbose=False)
                    self.history['train_loss'].append(train_loss)
                    self.history['train_metrics'].append(train_metrics)

                    if verbose:
                        print(f"Train Loss: {train_loss:.4f}")
                        if train_metrics:
                            for task, metrics in train_metrics.items():
                                print(f"  {task}: {metrics}")

                    # Validate
                    if val_loader is not None:
                        val_loss, val_metrics = self.validate(val_loader)
                        self.history['val_loss'].append(val_loss)
                        self.history['val_metrics'].append(val_metrics)

                        if verbose:
                            print(f"Val Loss: {val_loss:.4f}")
                            if val_metrics:
                                for task, metrics in val_metrics.items():
                                    print(f"  {task}: {metrics}")

                    # Log epoch summary
                    epoch_summary = {
                        'epoch': epoch + 1,
                        'train_loss': train_loss
                    }
                    if val_loader is not None:
                        epoch_summary['val_loss'] = val_loss

                    self.logger.info("Epoch completed", **epoch_summary)

                self.logger.info(
                    "Training completed successfully",
                    final_train_loss=self.history['train_loss'][-1],
                    final_val_loss=self.history['val_loss'][-1] if self.history['val_loss'] else None,
                    total_epochs=epochs
                )

                return self.history

        except Exception as e:
            self.logger.exception(
                "Training failed",
                error=str(e),
                epochs_completed=len(self.history['train_loss'])
            )
            raise

    def get_task_weights(self) -> Optional[Dict[str, float]]:
        """
        Get current task weights from the loss function.

        Returns:
            Dictionary of task weights or None
        """
        if hasattr(self.loss_fn, 'get_task_weights'):
            return self.loss_fn.get_task_weights()
        return None

    def get_task_uncertainties(self) -> Optional[Dict[str, float]]:
        """
        Get current task uncertainties (for uncertainty weighting).

        Returns:
            Dictionary of task uncertainties or None
        """
        if hasattr(self.loss_fn, 'get_uncertainties'):
            return self.loss_fn.get_uncertainties()
        return None


def create_multitask_dataloader(
    inputs: np.ndarray,
    targets: Dict[str, np.ndarray],
    batch_size: int = 32,
    shuffle: bool = True
) -> DataLoader:
    """
    Create a DataLoader for multi-task learning.

    Args:
        inputs: Input data (N, ...)
        targets: Dictionary mapping task names to target arrays (N, ...)
        batch_size: Batch size
        shuffle: Whether to shuffle data

    Returns:
        DataLoader yielding (inputs, targets_dict) tuples

    Example:
        >>> inputs = np.random.randn(1000, 784)
        >>> targets = {
        ...     'digit': np.random.randint(0, 10, 1000),
        ...     'parity': np.random.randint(0, 2, 1000)
        ... }
        >>> loader = create_multitask_dataloader(inputs, targets, batch_size=32)
    """
    from numpy_dl.data import TensorDataset, DataLoader
    from numpy_dl.core.tensor import Tensor

    # Convert to tensors
    input_tensor = Tensor(inputs)
    target_tensors = [Tensor(targets[task]) for task in sorted(targets.keys())]

    # Create dataset
    dataset = TensorDataset(input_tensor, *target_tensors)

    # Create data loader
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    # Wrap loader to return dict format
    class MultiTaskDataLoader:
        def __init__(self, loader, task_names):
            self.loader = loader
            self.task_names = task_names

        def __iter__(self):
            for batch in self.loader:
                inputs = batch[0]
                targets_dict = {
                    task: batch[i + 1]
                    for i, task in enumerate(self.task_names)
                }
                yield inputs, targets_dict

        def __len__(self):
            return len(self.loader)

    return MultiTaskDataLoader(loader, sorted(targets.keys()))
