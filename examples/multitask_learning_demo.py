"""
Comprehensive Multi-Task Learning Demo

This example demonstrates state-of-the-art multi-task learning techniques:
1. Hard parameter sharing with task-specific heads
2. Soft parameter sharing with cross-task regularization
3. Multiple loss weighting strategies (Uncertainty, GradNorm, DWA)
4. Multi-task metrics tracking and visualization

The example uses synthetic data with multiple related tasks to showcase
the benefits of multi-task learning.
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from numpy_dl.core.tensor import Tensor
from numpy_dl.models import (
    MLP,
    HardParameterSharing,
    SoftParameterSharing,
    TaskHead,
    create_hard_sharing_model,
    create_soft_sharing_model,
)
from numpy_dl.loss import (
    MSELoss,
    CrossEntropyLoss,
    MultiTaskLoss,
    UncertaintyWeighting,
    GradNorm,
    DynamicWeightAverage,
)
from numpy_dl.optim import Adam
from numpy_dl.utils import (
    accuracy,
    MultiTaskMetrics,
    MultiTaskTrainer,
    create_multitask_dataloader,
)
from numpy_dl.nn import ReLU


def generate_synthetic_multitask_data(n_samples=1000, n_features=20, noise_level=0.1):
    """
    Generate synthetic data for multi-task learning.

    Creates three related tasks:
    1. Regression: predict continuous value
    2. Binary classification: predict positive/negative
    3. Multi-class classification: predict one of 3 classes

    All tasks are based on the same underlying features with some shared structure.
    """
    np.random.seed(42)

    # Generate input features
    X = np.random.randn(n_samples, n_features).astype(np.float32)

    # Create shared representation (first 10 features are most informative)
    shared_features = X[:, :10]

    # Task 1: Regression - predict sum of first 5 features
    task1_target = np.sum(shared_features[:, :5], axis=1, keepdims=True)
    task1_target += noise_level * np.random.randn(n_samples, 1)
    task1_target = task1_target.astype(np.float32)

    # Task 2: Binary classification - positive if sum > 0
    task2_target = (np.sum(shared_features[:, 5:8], axis=1) > 0).astype(np.int64)

    # Task 3: Multi-class classification - based on feature relationships
    feature_sum = np.sum(shared_features[:, 8:10], axis=1)
    task3_target = np.zeros(n_samples, dtype=np.int64)
    task3_target[feature_sum < -1] = 0
    task3_target[(feature_sum >= -1) & (feature_sum < 1)] = 1
    task3_target[feature_sum >= 1] = 2

    return X, {
        'regression': task1_target,
        'binary': task2_target,
        'multiclass': task3_target,
    }


def accuracy_wrapper(preds, targets):
    """Wrapper for accuracy metric that handles predictions."""
    if preds.ndim > 1 and preds.shape[1] > 1:
        preds = np.argmax(preds, axis=1)
    elif preds.ndim > 1:
        preds = (preds > 0).astype(int).flatten()
    return accuracy(preds, targets)


def mse_metric(preds, targets):
    """Mean squared error metric."""
    return -float(np.mean((preds - targets) ** 2))  # Negative for "higher is better"


def demo_hard_parameter_sharing():
    """Demo 1: Hard parameter sharing with uncertainty weighting."""
    print("=" * 80)
    print("DEMO 1: Hard Parameter Sharing with Uncertainty Weighting")
    print("=" * 80)

    # Generate data
    print("\n1. Generating synthetic multi-task data...")
    X_train, y_train = generate_synthetic_multitask_data(n_samples=1000)
    X_val, y_val = generate_synthetic_multitask_data(n_samples=200)

    print(f"   Training samples: {X_train.shape[0]}")
    print(f"   Features: {X_train.shape[1]}")
    print(f"   Tasks: {list(y_train.keys())}")

    # Create dataloaders
    train_loader = create_multitask_dataloader(X_train, y_train, batch_size=32, shuffle=True)
    val_loader = create_multitask_dataloader(X_val, y_val, batch_size=32, shuffle=False)

    # Create model with hard parameter sharing
    print("\n2. Building hard parameter sharing model...")
    backbone = MLP(
        input_dim=20,
        hidden_dims=[128, 64],
        output_dim=32
    )

    task_configs = {
        'regression': {
            'output_dim': 1,
            'hidden_dims': [16],
        },
        'binary': {
            'output_dim': 2,
            'hidden_dims': [16],
            'activation': ReLU(),
        },
        'multiclass': {
            'output_dim': 3,
            'hidden_dims': [16],
            'activation': ReLU(),
        },
    }

    model = create_hard_sharing_model(backbone, task_configs)
    print(f"   Model: {model.__class__.__name__}")
    print(f"   Shared parameters: {sum(1 for _ in model.backbone.parameters())}")

    # Create loss function with uncertainty weighting
    print("\n3. Setting up uncertainty weighting loss...")
    task_losses = {
        'regression': MSELoss(),
        'binary': CrossEntropyLoss(),
        'multiclass': CrossEntropyLoss(),
    }

    task_types = {
        'regression': 'regression',
        'binary': 'classification',
        'multiclass': 'classification',
    }

    loss_fn = UncertaintyWeighting(task_losses, task_types)
    print(f"   Loss function: {loss_fn.__class__.__name__}")

    # Create optimizer
    optimizer = Adam(list(model.parameters()) + list(loss_fn.parameters()), lr=0.001)

    # Create trainer
    print("\n4. Training model...")
    metric_fns = {
        'regression': mse_metric,
        'binary': accuracy_wrapper,
        'multiclass': accuracy_wrapper,
    }

    trainer = MultiTaskTrainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device='cpu',
        metric_fns=metric_fns,
    )

    # Train
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=5,
        verbose=True
    )

    # Display learned uncertainties
    print("\n5. Learned task uncertainties:")
    uncertainties = trainer.get_task_uncertainties()
    weights = trainer.get_task_weights()
    for task in task_losses.keys():
        print(f"   {task:12s}: uncertainty={uncertainties[task]:.4f}, weight={weights[task]:.4f}")

    return history, model, loss_fn


def demo_soft_parameter_sharing():
    """Demo 2: Soft parameter sharing with regularization."""
    print("\n\n" + "=" * 80)
    print("DEMO 2: Soft Parameter Sharing with Cross-Task Regularization")
    print("=" * 80)

    # Generate data
    print("\n1. Generating data...")
    X_train, y_train = generate_synthetic_multitask_data(n_samples=1000)
    X_val, y_val = generate_synthetic_multitask_data(n_samples=200)

    train_loader = create_multitask_dataloader(X_train, y_train, batch_size=32, shuffle=True)
    val_loader = create_multitask_dataloader(X_val, y_val, batch_size=32, shuffle=False)

    # Create soft parameter sharing model
    print("\n2. Building soft parameter sharing model...")
    task_configs = {
        'regression': {
            'hidden_dims': [128, 64, 32],
            'output_dim': 1,
        },
        'binary': {
            'hidden_dims': [128, 64, 32],
            'output_dim': 2,
        },
        'multiclass': {
            'hidden_dims': [128, 64, 32],
            'output_dim': 3,
        },
    }

    model = create_soft_sharing_model(
        input_dim=20,
        task_configs=task_configs,
        sharing_strength=0.001  # Regularization strength
    )
    print(f"   Model: {model.__class__.__name__}")
    print(f"   Each task has its own network with parameter sharing regularization")

    # Create loss function with dynamic weight average
    print("\n3. Setting up Dynamic Weight Average loss...")
    task_losses = {
        'regression': MSELoss(),
        'binary': CrossEntropyLoss(),
        'multiclass': CrossEntropyLoss(),
    }

    loss_fn = DynamicWeightAverage(task_losses, temperature=2.0)
    print(f"   Loss function: {loss_fn.__class__.__name__}")

    # Create optimizer
    optimizer = Adam(model.parameters(), lr=0.001)

    # Create trainer
    print("\n4. Training model...")
    metric_fns = {
        'regression': mse_metric,
        'binary': accuracy_wrapper,
        'multiclass': accuracy_wrapper,
    }

    trainer = MultiTaskTrainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device='cpu',
        metric_fns=metric_fns,
    )

    # Train
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=5,
        verbose=True
    )

    # Display final task weights
    print("\n5. Final dynamic task weights:")
    weights = trainer.get_task_weights()
    for task, weight in weights.items():
        print(f"   {task:12s}: {weight:.4f}")

    # Display sharing loss
    sharing_loss = model.compute_sharing_loss()
    if sharing_loss is not None:
        print(f"\n6. Parameter sharing regularization loss: {float(sharing_loss.data):.6f}")

    return history, model, loss_fn


def demo_loss_weighting_comparison():
    """Demo 3: Compare different loss weighting strategies."""
    print("\n\n" + "=" * 80)
    print("DEMO 3: Comparing Loss Weighting Strategies")
    print("=" * 80)

    # Generate data
    X_train, y_train = generate_synthetic_multitask_data(n_samples=1000)
    X_val, y_val = generate_synthetic_multitask_data(n_samples=200)

    train_loader = create_multitask_dataloader(X_train, y_train, batch_size=32, shuffle=True)
    val_loader = create_multitask_dataloader(X_val, y_val, batch_size=32, shuffle=False)

    # Task losses and metrics
    task_losses = {
        'regression': MSELoss(),
        'binary': CrossEntropyLoss(),
        'multiclass': CrossEntropyLoss(),
    }

    metric_fns = {
        'regression': mse_metric,
        'binary': accuracy_wrapper,
        'multiclass': accuracy_wrapper,
    }

    # Test different weighting strategies
    strategies = {
        'Equal Weighting': MultiTaskLoss(task_losses),
        'Uncertainty Weighting': UncertaintyWeighting(
            task_losses,
            {'regression': 'regression', 'binary': 'classification', 'multiclass': 'classification'}
        ),
        'Dynamic Weight Average': DynamicWeightAverage(task_losses, temperature=2.0),
    }

    results = {}

    for strategy_name, loss_fn in strategies.items():
        print(f"\n{'=' * 40}")
        print(f"Testing: {strategy_name}")
        print(f"{'=' * 40}")

        # Create model
        backbone = MLP(input_dim=20, hidden_dims=[128, 64], output_dim=32)
        task_configs = {
            'regression': {'output_dim': 1, 'hidden_dims': [16]},
            'binary': {'output_dim': 2, 'hidden_dims': [16]},
            'multiclass': {'output_dim': 3, 'hidden_dims': [16]},
        }
        model = create_hard_sharing_model(backbone, task_configs)

        # Create optimizer
        params = list(model.parameters())
        if hasattr(loss_fn, 'log_vars') or hasattr(loss_fn, 'weight_params'):
            params += list(loss_fn.parameters())
        optimizer = Adam(params, lr=0.001)

        # Create trainer
        trainer = MultiTaskTrainer(
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device='cpu',
            metric_fns=metric_fns,
        )

        # Train
        history = trainer.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=3,
            verbose=False
        )

        # Store results
        final_metrics = history['val_metrics'][-1]
        results[strategy_name] = final_metrics

        print(f"\nFinal validation metrics:")
        for task, metrics in final_metrics.items():
            print(f"  {task:12s}: {metrics}")

    # Compare results
    print("\n\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)
    print("\nAverage task performance across strategies:")
    for strategy_name, metrics in results.items():
        avg_metric = metrics.get('average', {}).get('metric', 0)
        print(f"  {strategy_name:25s}: {avg_metric:.4f}")

    return results


def demo_custom_architecture():
    """Demo 4: Custom multi-task architecture with flexible task heads."""
    print("\n\n" + "=" * 80)
    print("DEMO 4: Custom Architecture with Flexible Task Heads")
    print("=" * 80)

    print("\n1. Building custom architecture...")

    # Create a custom backbone (e.g., deeper network)
    backbone = MLP(
        input_dim=20,
        hidden_dims=[256, 128, 64],
        output_dim=64
    )

    # Create task heads with different architectures
    task_heads = {
        'regression': TaskHead(
            input_dim=64,
            output_dim=1,
            hidden_dims=[32, 16],  # Deeper head
            activation=ReLU(),
            dropout=0.1
        ),
        'binary': TaskHead(
            input_dim=64,
            output_dim=2,
            hidden_dims=[32],  # Shallower head
            activation=ReLU(),
        ),
        'multiclass': TaskHead(
            input_dim=64,
            output_dim=3,
            hidden_dims=[48, 24],  # Different width
            activation=ReLU(),
            dropout=0.2
        ),
    }

    model = HardParameterSharing(
        backbone=backbone,
        task_heads=task_heads,
        shared_head_dim=None  # No additional shared head
    )

    print(f"   Backbone: {len(list(backbone.parameters()))} parameter groups")
    print(f"   Task heads:")
    for task_name, head in task_heads.items():
        n_params = sum(np.prod(p.data.shape) for p in head.parameters())
        print(f"     {task_name:12s}: {n_params:6d} parameters")

    # Quick test
    print("\n2. Testing forward pass...")
    test_input = Tensor(np.random.randn(8, 20).astype(np.float32))
    outputs = model(test_input)

    print(f"   Input shape: {test_input.shape}")
    for task_name, output in outputs.items():
        print(f"   {task_name:12s} output shape: {output.shape}")

    print("\n   Custom architecture created successfully!")

    return model


def main():
    """Run all multi-task learning demos."""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 15 + "MULTI-TASK LEARNING COMPREHENSIVE DEMO" + " " * 24 + "║")
    print("║" + " " * 78 + "║")
    print("║  This demo showcases state-of-the-art multi-task learning techniques:" + " " * 7 + "║")
    print("║  - Hard and soft parameter sharing architectures" + " " * 28 + "║")
    print("║  - Uncertainty weighting, GradNorm, and Dynamic Weight Average" + " " * 15 + "║")
    print("║  - Multi-task metrics tracking and comparison" + " " * 31 + "║")
    print("║  - Custom architectures with flexible task heads" + " " * 28 + "║")
    print("╚" + "=" * 78 + "╝")

    try:
        # Run demos
        history1, model1, loss1 = demo_hard_parameter_sharing()
        history2, model2, loss2 = demo_soft_parameter_sharing()
        results = demo_loss_weighting_comparison()
        model_custom = demo_custom_architecture()

        print("\n\n" + "=" * 80)
        print("ALL DEMOS COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print("\nKey Takeaways:")
        print("  1. Hard parameter sharing is efficient and works well for related tasks")
        print("  2. Soft parameter sharing provides more flexibility at higher computational cost")
        print("  3. Uncertainty weighting automatically balances task losses")
        print("  4. Dynamic Weight Average adapts to changing task difficulties")
        print("  5. Custom architectures allow task-specific design choices")
        print("\nThe framework is:")
        print("  ✓ Extendible: Easy to add new loss weighting strategies")
        print("  ✓ Composable: Mix and match architectures and loss functions")
        print("  ✓ Debuggable: Clear APIs and comprehensive metrics tracking")

    except Exception as e:
        print(f"\n\nError during demo: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
