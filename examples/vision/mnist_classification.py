"""
MNIST Classification Example

Train a simple CNN on the MNIST dataset for digit classification.
"""

import numpy as np
import numpy_dl as ndl
from numpy_dl.models import SimpleCNN
from numpy_dl.optim import Adam
from numpy_dl.loss import CrossEntropyLoss
from numpy_dl.data import TensorDataset, DataLoader, train_test_split
from numpy_dl.utils import accuracy, MetricTracker, plot_training_history
from numpy_dl.tracking import ExperimentTracker


def load_mnist_data():
    """Load MNIST dataset (placeholder - in practice, would load from file)."""
    # Generate synthetic data for demonstration
    print("Generating synthetic MNIST-like data...")
    n_train = 1000
    n_test = 200

    X_train = np.random.randn(n_train, 1, 28, 28).astype(np.float32)
    y_train = np.random.randint(0, 10, n_train)

    X_test = np.random.randn(n_test, 1, 28, 28).astype(np.float32)
    y_test = np.random.randint(0, 10, n_test)

    return X_train, y_train, X_test, y_test


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    metric_tracker = MetricTracker()

    for batch_idx, (data, target) in enumerate(dataloader):
        # Convert to tensors
        data = ndl.tensor(data, requires_grad=False, device=device)
        target = ndl.tensor(target, device=device)

        # Forward pass
        output = model(data)
        loss = criterion(output, target)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track metrics
        acc = accuracy(output.numpy(), target.numpy())
        metric_tracker.update(loss=loss.item(), accuracy=acc)

        if batch_idx % 10 == 0:
            print(f"  Batch {batch_idx}/{len(dataloader)}, "
                  f"Loss: {loss.item():.4f}, Acc: {acc:.4f}")

    return metric_tracker.compute()


def evaluate(model, dataloader, criterion, device):
    """Evaluate model."""
    model.eval()
    metric_tracker = MetricTracker()

    for data, target in dataloader:
        data = ndl.tensor(data, requires_grad=False, device=device)
        target = ndl.tensor(target, device=device)

        output = model(data)
        loss = criterion(output, target)

        acc = accuracy(output.numpy(), target.numpy())
        metric_tracker.update(loss=loss.item(), accuracy=acc)

    return metric_tracker.compute()


def main():
    """Main training function."""
    # Hyperparameters
    batch_size = 32
    epochs = 5
    learning_rate = 0.001
    device = 'cpu'

    # Initialize experiment tracker
    tracker = ExperimentTracker('mnist_cnn')
    tracker.log_hyperparameters(
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        model='SimpleCNN'
    )

    # Load data
    X_train, y_train, X_test, y_test = load_mnist_data()

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Create model
    model = SimpleCNN(
        input_channels=1,
        num_classes=10,
        conv_channels=[16, 32],
        fc_sizes=[128],
        input_size=(28, 28),
        dropout=0.5
    )
    model.to(device)

    # Loss and optimizer
    criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    print(f"Model: {model}")
    print(f"Device: {device}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    # Training loop
    history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")

        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device)
        print(f"Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['accuracy']:.4f}")

        # Evaluate
        test_metrics = evaluate(model, test_loader, criterion, device)
        print(f"Test Loss: {test_metrics['loss']:.4f}, Test Acc: {test_metrics['accuracy']:.4f}")

        # Track metrics
        history['train_loss'].append(train_metrics['loss'])
        history['train_acc'].append(train_metrics['accuracy'])
        history['test_loss'].append(test_metrics['loss'])
        history['test_acc'].append(test_metrics['accuracy'])

        tracker.log_metrics(
            epoch=epoch,
            train_loss=train_metrics['loss'],
            train_accuracy=train_metrics['accuracy'],
            test_loss=test_metrics['loss'],
            test_accuracy=test_metrics['accuracy']
        )

        # Save checkpoint
        if (epoch + 1) % 2 == 0:
            tracker.log_model_checkpoint(
                model.state_dict(),
                epoch=epoch,
                metrics={'test_accuracy': test_metrics['accuracy']}
            )

    # Finish experiment
    tracker.finish()

    # Plot results
    print("\nPlotting training history...")
    plot_training_history(history)

    print("\nTraining complete!")


if __name__ == '__main__':
    main()
