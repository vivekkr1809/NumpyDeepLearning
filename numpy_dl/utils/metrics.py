"""Metrics for evaluating model performance."""

import numpy as np
from typing import Union, List


def accuracy(predictions: np.ndarray, targets: np.ndarray) -> float:
    """
    Compute classification accuracy.

    Args:
        predictions: Predicted class indices or logits
        targets: True class indices

    Returns:
        Accuracy score
    """
    if predictions.ndim > 1:
        predictions = np.argmax(predictions, axis=-1)

    return np.mean(predictions == targets)


def precision(predictions: np.ndarray, targets: np.ndarray, average: str = 'macro') -> float:
    """
    Compute precision score.

    Args:
        predictions: Predicted class indices
        targets: True class indices
        average: Averaging strategy ('macro', 'micro', 'weighted')

    Returns:
        Precision score
    """
    if predictions.ndim > 1:
        predictions = np.argmax(predictions, axis=-1)

    classes = np.unique(targets)
    precisions = []

    for cls in classes:
        true_positive = np.sum((predictions == cls) & (targets == cls))
        false_positive = np.sum((predictions == cls) & (targets != cls))

        if true_positive + false_positive > 0:
            precisions.append(true_positive / (true_positive + false_positive))
        else:
            precisions.append(0.0)

    if average == 'macro':
        return np.mean(precisions)
    elif average == 'micro':
        tp = np.sum(predictions == targets)
        fp = np.sum(predictions != targets)
        return tp / (tp + fp) if (tp + fp) > 0 else 0.0
    elif average == 'weighted':
        weights = [np.sum(targets == cls) for cls in classes]
        return np.average(precisions, weights=weights)
    else:
        return np.array(precisions)


def recall(predictions: np.ndarray, targets: np.ndarray, average: str = 'macro') -> float:
    """
    Compute recall score.

    Args:
        predictions: Predicted class indices
        targets: True class indices
        average: Averaging strategy ('macro', 'micro', 'weighted')

    Returns:
        Recall score
    """
    if predictions.ndim > 1:
        predictions = np.argmax(predictions, axis=-1)

    classes = np.unique(targets)
    recalls = []

    for cls in classes:
        true_positive = np.sum((predictions == cls) & (targets == cls))
        false_negative = np.sum((predictions != cls) & (targets == cls))

        if true_positive + false_negative > 0:
            recalls.append(true_positive / (true_positive + false_negative))
        else:
            recalls.append(0.0)

    if average == 'macro':
        return np.mean(recalls)
    elif average == 'micro':
        return accuracy(predictions, targets)
    elif average == 'weighted':
        weights = [np.sum(targets == cls) for cls in classes]
        return np.average(recalls, weights=weights)
    else:
        return np.array(recalls)


def f1_score(predictions: np.ndarray, targets: np.ndarray, average: str = 'macro') -> float:
    """
    Compute F1 score.

    Args:
        predictions: Predicted class indices
        targets: True class indices
        average: Averaging strategy ('macro', 'micro', 'weighted')

    Returns:
        F1 score
    """
    prec = precision(predictions, targets, average=average)
    rec = recall(predictions, targets, average=average)

    if isinstance(prec, np.ndarray):
        f1 = np.zeros_like(prec)
        mask = (prec + rec) > 0
        f1[mask] = 2 * prec[mask] * rec[mask] / (prec[mask] + rec[mask])
        return f1
    else:
        if prec + rec > 0:
            return 2 * prec * rec / (prec + rec)
        return 0.0


def confusion_matrix(predictions: np.ndarray, targets: np.ndarray, num_classes: int = None) -> np.ndarray:
    """
    Compute confusion matrix.

    Args:
        predictions: Predicted class indices
        targets: True class indices
        num_classes: Number of classes (inferred if None)

    Returns:
        Confusion matrix of shape (num_classes, num_classes)
    """
    if predictions.ndim > 1:
        predictions = np.argmax(predictions, axis=-1)

    if num_classes is None:
        num_classes = max(np.max(predictions), np.max(targets)) + 1

    cm = np.zeros((num_classes, num_classes), dtype=int)

    for pred, target in zip(predictions.flatten(), targets.flatten()):
        cm[int(target), int(pred)] += 1

    return cm


def mean_squared_error(predictions: np.ndarray, targets: np.ndarray) -> float:
    """
    Compute mean squared error.

    Args:
        predictions: Predicted values
        targets: True values

    Returns:
        MSE score
    """
    return np.mean((predictions - targets) ** 2)


def mean_absolute_error(predictions: np.ndarray, targets: np.ndarray) -> float:
    """
    Compute mean absolute error.

    Args:
        predictions: Predicted values
        targets: True values

    Returns:
        MAE score
    """
    return np.mean(np.abs(predictions - targets))


def r2_score(predictions: np.ndarray, targets: np.ndarray) -> float:
    """
    Compute R² score.

    Args:
        predictions: Predicted values
        targets: True values

    Returns:
        R² score
    """
    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)

    if ss_tot == 0:
        return 0.0

    return 1 - (ss_res / ss_tot)


class MetricTracker:
    """
    Track metrics during training.

    Accumulates metrics over batches and epochs.
    """

    def __init__(self):
        """Initialize metric tracker."""
        self.metrics = {}
        self.history = {}

    def update(self, **metrics):
        """
        Update metrics.

        Args:
            **metrics: Metric name-value pairs
        """
        for name, value in metrics.items():
            if name not in self.metrics:
                self.metrics[name] = []
            self.metrics[name].append(float(value))

    def compute(self, reset: bool = True) -> dict:
        """
        Compute average of accumulated metrics.

        Args:
            reset: Whether to reset metrics after computing

        Returns:
            Dictionary of averaged metrics
        """
        result = {}
        for name, values in self.metrics.items():
            if values:
                result[name] = np.mean(values)

        if reset:
            self.save_to_history()
            self.reset()

        return result

    def reset(self):
        """Reset all metrics."""
        self.metrics = {}

    def save_to_history(self):
        """Save current metrics to history."""
        for name, values in self.metrics.items():
            if name not in self.history:
                self.history[name] = []
            if values:
                self.history[name].append(np.mean(values))

    def get_history(self, metric_name: str = None) -> Union[List, dict]:
        """
        Get metric history.

        Args:
            metric_name: Name of metric (returns all if None)

        Returns:
            List of values or dictionary of all histories
        """
        if metric_name is None:
            return self.history
        return self.history.get(metric_name, [])
