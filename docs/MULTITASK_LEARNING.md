# Multi-Task Learning in NumPy Deep Learning

This document provides a comprehensive guide to the multi-task learning (MTL) capabilities in the NumPy Deep Learning framework.

## Table of Contents

1. [Overview](#overview)
2. [Architecture Patterns](#architecture-patterns)
3. [Loss Weighting Strategies](#loss-weighting-strategies)
4. [Quick Start](#quick-start)
5. [API Reference](#api-reference)
6. [Examples](#examples)
7. [Best Practices](#best-practices)

## Overview

Multi-task learning (MTL) is a machine learning paradigm where multiple related tasks are learned simultaneously, leveraging shared representations to improve generalization and efficiency.

### Benefits of Multi-Task Learning

- **Improved Generalization**: Learning multiple tasks simultaneously provides an inductive bias that helps prevent overfitting
- **Parameter Efficiency**: Shared representations reduce the total number of parameters needed
- **Faster Training**: Related tasks can benefit from each other's learning signals
- **Better Sample Efficiency**: Tasks with more data can help tasks with less data

### Supported Features

This framework implements state-of-the-art MTL techniques from recent research (2018-2025):

#### Architecture Patterns
- **Hard Parameter Sharing**: Shared backbone with task-specific heads
- **Soft Parameter Sharing**: Separate networks with cross-task regularization
- **Multi-Head Architecture**: Flexible wrapper for custom designs

#### Loss Weighting Strategies
- **Equal Weighting**: Baseline approach (uniform weights)
- **Uncertainty Weighting**: Learns task uncertainties to balance losses (Kendall et al., CVPR 2018)
- **GradNorm**: Balances gradient magnitudes across tasks (Chen et al., ICML 2018)
- **Dynamic Weight Average (DWA)**: Adapts weights based on learning progress (Liu et al., CVPR 2019)

## Architecture Patterns

### Hard Parameter Sharing

The most common MTL architecture where tasks share a common backbone network with task-specific heads.

```
Input → Shared Backbone → Task Head 1 → Output 1
                       ├→ Task Head 2 → Output 2
                       └→ Task Head N → Output N
```

**Advantages:**
- Parameter efficient
- Strong regularization effect
- Fast training

**Disadvantages:**
- Less flexibility per task
- Risk of negative transfer if tasks are too different

**Example:**

```python
from numpy_dl.models import MLP, create_hard_sharing_model
from numpy_dl.nn import ReLU

# Create shared backbone
backbone = MLP(input_dim=784, hidden_dims=[512, 256], output_dim=128)

# Define task configurations
task_configs = {
    'task1': {'output_dim': 10, 'hidden_dims': [64]},
    'task2': {'output_dim': 2, 'activation': ReLU()}
}

# Create model
model = create_hard_sharing_model(backbone, task_configs)

# Forward pass
outputs = model(input_data)  # Returns: {'task1': out1, 'task2': out2}
```

### Soft Parameter Sharing

Each task has its own network, but networks are encouraged to share similar parameters through regularization.

```
Input → Task Network 1 → Output 1
     ├→ Task Network 2 → Output 2
     └→ Task Network N → Output N
         (with parameter similarity regularization)
```

**Advantages:**
- More flexibility per task
- Can handle more diverse tasks
- Reduces risk of negative transfer

**Disadvantages:**
- More parameters
- Slower training
- Requires careful tuning of regularization strength

**Example:**

```python
from numpy_dl.models import create_soft_sharing_model

task_configs = {
    'task1': {'hidden_dims': [256, 128], 'output_dim': 10},
    'task2': {'hidden_dims': [256, 128], 'output_dim': 2}
}

model = create_soft_sharing_model(
    input_dim=784,
    task_configs=task_configs,
    sharing_strength=0.01  # Regularization strength
)

outputs = model(input_data)
```

## Loss Weighting Strategies

### Equal Weighting (Baseline)

Simply sums all task losses with equal weight.

```python
from numpy_dl.loss import MultiTaskLoss, MSELoss, CrossEntropyLoss

task_losses = {
    'regression': MSELoss(),
    'classification': CrossEntropyLoss()
}

loss_fn = MultiTaskLoss(task_losses)
```

### Uncertainty Weighting

Automatically learns task-dependent uncertainties to balance losses. Each task has a learnable log-variance parameter.

**Formula:**
```
L_weighted = (1 / (2σ²)) * L_task + log(σ)
```

**When to use:**
- Tasks have different output scales
- Mixing regression and classification
- Want automatic task balancing

**Example:**

```python
from numpy_dl.loss import UncertaintyWeighting

task_losses = {
    'depth': MSELoss(),
    'segmentation': CrossEntropyLoss()
}

task_types = {
    'depth': 'regression',
    'segmentation': 'classification'
}

loss_fn = UncertaintyWeighting(task_losses, task_types)

# After training, check learned uncertainties
uncertainties = loss_fn.get_uncertainties()
weights = loss_fn.get_task_weights()
```

### GradNorm

Balances training by normalizing gradient magnitudes across tasks.

**When to use:**
- Tasks are learning at different rates
- Want to ensure all tasks make progress
- Need explicit gradient balancing

**Example:**

```python
from numpy_dl.loss import GradNorm

loss_fn = GradNorm(task_losses, alpha=1.5)

# Get current task weights
weights = loss_fn.get_task_weights()
```

### Dynamic Weight Average (DWA)

Adjusts task weights based on the rate of change of task losses. Tasks with faster decreasing losses get lower weights.

**When to use:**
- Task difficulties change during training
- Want adaptive balancing
- Tasks have varying convergence rates

**Example:**

```python
from numpy_dl.loss import DynamicWeightAverage

loss_fn = DynamicWeightAverage(
    task_losses,
    temperature=2.0,  # Controls weight adaptation speed
    window_size=2     # Lookback window for loss rate
)
```

## Quick Start

### Basic Multi-Task Training

```python
import numpy as np
from numpy_dl.models import MLP, create_hard_sharing_model
from numpy_dl.loss import UncertaintyWeighting, MSELoss, CrossEntropyLoss
from numpy_dl.optim import Adam
from numpy_dl.utils import MultiTaskTrainer, create_multitask_dataloader

# 1. Prepare data
X_train = np.random.randn(1000, 20).astype(np.float32)
y_train = {
    'task1': np.random.randn(1000, 1).astype(np.float32),
    'task2': np.random.randint(0, 5, 1000)
}

train_loader = create_multitask_dataloader(X_train, y_train, batch_size=32)

# 2. Create model
backbone = MLP(20, [128, 64], 32)
task_configs = {
    'task1': {'output_dim': 1},
    'task2': {'output_dim': 5}
}
model = create_hard_sharing_model(backbone, task_configs)

# 3. Create loss function
task_losses = {
    'task1': MSELoss(),
    'task2': CrossEntropyLoss()
}
task_types = {'task1': 'regression', 'task2': 'classification'}
loss_fn = UncertaintyWeighting(task_losses, task_types)

# 4. Create optimizer (include loss parameters for uncertainty weighting)
optimizer = Adam(list(model.parameters()) + list(loss_fn.parameters()), lr=0.001)

# 5. Create trainer
trainer = MultiTaskTrainer(
    model=model,
    loss_fn=loss_fn,
    optimizer=optimizer,
    device='cpu'
)

# 6. Train
history = trainer.fit(train_loader, epochs=10)

# 7. Check results
print("Learned uncertainties:", loss_fn.get_uncertainties())
print("Task weights:", loss_fn.get_task_weights())
```

## API Reference

### Models

#### `TaskHead(input_dim, output_dim, hidden_dims=None, activation=None, dropout=0.0)`

Task-specific head for multi-task models.

**Parameters:**
- `input_dim` (int): Input feature dimension from shared backbone
- `output_dim` (int): Output dimension for the task
- `hidden_dims` (list): List of hidden layer dimensions
- `activation` (Module): Activation function
- `dropout` (float): Dropout rate

#### `HardParameterSharing(backbone, task_heads, shared_head_dim=None)`

Hard parameter sharing architecture.

**Parameters:**
- `backbone` (Module): Shared backbone network
- `task_heads` (dict): Dictionary mapping task names to TaskHead modules
- `shared_head_dim` (int, optional): Dimension for additional shared head layer

#### `SoftParameterSharing(task_networks, sharing_strength=0.01, sharing_groups=None)`

Soft parameter sharing architecture.

**Parameters:**
- `task_networks` (dict): Dictionary mapping task names to networks
- `sharing_strength` (float): Regularization strength for parameter sharing
- `sharing_groups` (list, optional): List of task groups for grouped regularization

#### `create_hard_sharing_model(backbone, task_configs, shared_head_dim=None)`

Factory function to create a hard parameter sharing model.

**Parameters:**
- `backbone` (Module): Shared backbone network
- `task_configs` (dict): Dictionary of task configurations
- `shared_head_dim` (int, optional): Shared head dimension

### Loss Functions

#### `MultiTaskLoss(task_losses, loss_weights=None, reduction='mean')`

Base multi-task loss with equal or fixed weighting.

#### `UncertaintyWeighting(task_losses, task_types=None, init_log_vars=None, reduction='mean')`

Uncertainty-based loss weighting.

**Methods:**
- `get_task_weights()`: Get current task weights
- `get_uncertainties()`: Get current task uncertainties (sigma values)

#### `GradNorm(task_losses, alpha=1.5, reduction='mean')`

Gradient normalization for loss balancing.

**Parameters:**
- `alpha` (float): Hyperparameter controlling adaptation rate

#### `DynamicWeightAverage(task_losses, temperature=2.0, window_size=2, reduction='mean')`

Dynamic weight average based on loss rate of change.

**Parameters:**
- `temperature` (float): Temperature for softmax weighting
- `window_size` (int): Number of previous losses to consider

### Training Utilities

#### `MultiTaskTrainer(model, loss_fn, optimizer, device='cpu', metric_fns=None, grad_clip=None)`

Trainer for multi-task learning models.

**Methods:**
- `train_epoch(train_loader, verbose=True)`: Train for one epoch
- `validate(val_loader)`: Validate the model
- `fit(train_loader, val_loader=None, epochs=10, verbose=True)`: Train for multiple epochs
- `get_task_weights()`: Get current task weights from loss function
- `get_task_uncertainties()`: Get current task uncertainties

#### `MultiTaskMetrics(task_names, metric_fns)`

Track and compute metrics for multiple tasks.

**Methods:**
- `update(predictions, targets, losses=None)`: Update with batch results
- `compute()`: Compute final metrics for all tasks
- `compute_task_metric(task)`: Compute metric for specific task
- `reset()`: Reset all metrics

#### `create_multitask_dataloader(inputs, targets, batch_size=32, shuffle=True)`

Create a DataLoader for multi-task learning.

**Parameters:**
- `inputs` (ndarray): Input data (N, ...)
- `targets` (dict): Dictionary mapping task names to target arrays
- `batch_size` (int): Batch size
- `shuffle` (bool): Whether to shuffle data

## Examples

### Example 1: Multi-Task MNIST

Train a model to simultaneously predict digit class and digit parity (odd/even):

```python
from numpy_dl.models import SimpleCNN, TaskHead, HardParameterSharing
from numpy_dl.loss import UncertaintyWeighting, CrossEntropyLoss
from numpy_dl.nn import ReLU

# Create CNN backbone
backbone = SimpleCNN(
    in_channels=1,
    num_classes=128,  # Feature dimension
    img_size=28
)

# Task heads
task_heads = {
    'digit': TaskHead(128, 10, hidden_dims=[64], activation=ReLU()),
    'parity': TaskHead(128, 2, hidden_dims=[32], activation=ReLU())
}

model = HardParameterSharing(backbone, task_heads)

# Loss function
task_losses = {
    'digit': CrossEntropyLoss(),
    'parity': CrossEntropyLoss()
}
task_types = {
    'digit': 'classification',
    'parity': 'classification'
}

loss_fn = UncertaintyWeighting(task_losses, task_types)

# Train as before...
```

### Example 2: Regression + Classification

Combine regression and classification tasks:

```python
# Generate synthetic data
X = np.random.randn(1000, 50)
y_regression = X[:, :10].sum(axis=1, keepdims=True)  # Continuous
y_classification = (X[:, 10:20].sum(axis=1) > 0).astype(int)  # Binary

targets = {
    'regression': y_regression,
    'classification': y_classification
}

# Create model
backbone = MLP(50, [256, 128], 64)
task_configs = {
    'regression': {'output_dim': 1},
    'classification': {'output_dim': 2}
}
model = create_hard_sharing_model(backbone, task_configs)

# Use uncertainty weighting for automatic balancing
task_losses = {
    'regression': MSELoss(),
    'classification': CrossEntropyLoss()
}
task_types = {
    'regression': 'regression',
    'classification': 'classification'
}
loss_fn = UncertaintyWeighting(task_losses, task_types)
```

### Example 3: Three Related Tasks

```python
# Prepare data with three related tasks
targets = {
    'task_a': y_a,  # Classification (10 classes)
    'task_b': y_b,  # Classification (5 classes)
    'task_c': y_c,  # Regression
}

# Create model
backbone = MLP(input_dim, [512, 256, 128], 128)
task_configs = {
    'task_a': {'output_dim': 10, 'hidden_dims': [64, 32]},
    'task_b': {'output_dim': 5, 'hidden_dims': [64]},
    'task_c': {'output_dim': 1, 'hidden_dims': [64, 32]}
}
model = create_hard_sharing_model(backbone, task_configs)

# Use Dynamic Weight Average
task_losses = {
    'task_a': CrossEntropyLoss(),
    'task_b': CrossEntropyLoss(),
    'task_c': MSELoss()
}
loss_fn = DynamicWeightAverage(task_losses, temperature=2.0)

# Setup metrics
from numpy_dl.utils import accuracy
metric_fns = {
    'task_a': accuracy,
    'task_b': accuracy,
    'task_c': lambda p, t: -np.mean((p - t) ** 2)  # Negative MSE
}

trainer = MultiTaskTrainer(model, loss_fn, optimizer, metric_fns=metric_fns)
history = trainer.fit(train_loader, val_loader, epochs=20)
```

## Best Practices

### Choosing an Architecture

1. **Use Hard Parameter Sharing** when:
   - Tasks are closely related
   - You need parameter efficiency
   - You want strong regularization
   - Tasks have similar input-output structures

2. **Use Soft Parameter Sharing** when:
   - Tasks are moderately related
   - You have sufficient computational resources
   - Risk of negative transfer is high
   - Tasks require different feature representations

### Choosing a Loss Weighting Strategy

1. **Equal Weighting**: Start here as a baseline, especially if tasks have similar scales

2. **Uncertainty Weighting**: Best for:
   - Mixing regression and classification
   - Tasks with very different loss magnitudes
   - Automatic task balancing without manual tuning

3. **GradNorm**: Best for:
   - Ensuring balanced learning progress
   - When gradient magnitudes differ significantly
   - Need explicit control over gradient balancing

4. **Dynamic Weight Average**: Best for:
   - Tasks with varying difficulty over training
   - Adaptive task balancing
   - When task convergence rates differ

### Training Tips

1. **Start Simple**: Begin with equal weighting and hard parameter sharing

2. **Monitor Task Performance**: Track individual task metrics to detect negative transfer

3. **Learning Rates**: Consider using different learning rates for shared vs. task-specific parameters

4. **Gradient Clipping**: Use gradient clipping to stabilize training, especially with uncertainty weighting

5. **Validation**: Use validation data to tune hyperparameters (temperature, alpha, sharing_strength)

6. **Task Relationships**: Group related tasks together when using soft parameter sharing

### Debugging

1. **Check Task Weights**: Monitor learned weights to ensure no task dominates

2. **Individual Task Training**: Train tasks separately to establish baseline performance

3. **Visualize Loss History**: Plot individual task losses over time

4. **Gradient Norms**: Check gradient norms per task to detect training issues

## References

1. **Caruana, R. (1997).** "Multitask Learning." Machine Learning, 28(1), 41-75.

2. **Kendall, A., Gal, Y., & Cipolla, R. (2018).** "Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics." CVPR 2018.

3. **Chen, Z., Badrinarayanan, V., Lee, C. Y., & Rabinovich, A. (2018).** "GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks." ICML 2018.

4. **Liu, S., Johns, E., & Davison, A. J. (2019).** "End-to-End Multi-Task Learning with Attention." CVPR 2019.

5. **Crawshaw, M. (2020).** "Multi-Task Learning with Deep Neural Networks: A Survey." arXiv:2009.09796.

6. **Ruder, S. (2017).** "An Overview of Multi-Task Learning in Deep Neural Networks." arXiv:1706.05098.

## Contributing

To extend the multi-task learning functionality:

1. **New Loss Weighting Strategy**: Subclass `MultiTaskLoss` and implement custom weighting logic

2. **New Architecture Pattern**: Create a new module that follows the `forward(x) -> Dict[str, Tensor]` interface

3. **New Metrics**: Implement functions with signature `(predictions, targets) -> float`

Example custom loss weighting:

```python
from numpy_dl.loss.multitask import MultiTaskLoss

class CustomWeighting(MultiTaskLoss):
    def __init__(self, task_losses, custom_param=1.0):
        super().__init__(task_losses)
        self.custom_param = custom_param

    def forward(self, outputs, targets, return_dict=False):
        task_losses = self.compute_task_losses(outputs, targets)

        # Implement custom weighting logic
        weighted_losses = {}
        for task_name, loss in task_losses.items():
            weight = self.compute_custom_weight(task_name)
            weighted_losses[task_name] = weight * loss

        total_loss = sum(weighted_losses.values())

        if return_dict:
            return total_loss, task_losses
        return total_loss

    def compute_custom_weight(self, task_name):
        # Your custom weighting logic here
        return 1.0
```

---

**Framework Properties:**
- ✅ **Extendible**: Easy to add new architectures and loss strategies
- ✅ **Composable**: Mix and match components freely
- ✅ **Debuggable**: Clear APIs and comprehensive logging
