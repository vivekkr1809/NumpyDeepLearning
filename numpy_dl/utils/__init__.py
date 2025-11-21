"""Utility functions and classes."""

from numpy_dl.utils.device import (
    Device, get_array_module, to_device,
    get_default_device, set_default_device, cuda_is_available
)
from numpy_dl.utils.config import Config, create_default_config
from numpy_dl.utils.metrics import (
    accuracy, precision, recall, f1_score, confusion_matrix,
    mean_squared_error, mean_absolute_error, r2_score, MetricTracker
)
from numpy_dl.utils.visualization import (
    plot_training_history, plot_confusion_matrix, plot_predictions,
    plot_images_grid, plot_feature_maps, plot_gradient_flow
)
from numpy_dl.utils.multitask import (
    MultiTaskMetrics,
    MultiTaskTrainer,
    create_multitask_dataloader,
)

__all__ = [
    'Device',
    'get_array_module',
    'to_device',
    'get_default_device',
    'set_default_device',
    'cuda_is_available',
    'Config',
    'create_default_config',
    'accuracy',
    'precision',
    'recall',
    'f1_score',
    'confusion_matrix',
    'mean_squared_error',
    'mean_absolute_error',
    'r2_score',
    'MetricTracker',
    'plot_training_history',
    'plot_confusion_matrix',
    'plot_predictions',
    'plot_images_grid',
    'plot_feature_maps',
    'plot_gradient_flow',
    # Multi-task learning
    'MultiTaskMetrics',
    'MultiTaskTrainer',
    'create_multitask_dataloader',
]
