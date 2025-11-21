"""
Basic Autoencoder for MNIST Reconstruction.

This example demonstrates:
1. Training a standard fully-connected autoencoder
2. Visualizing reconstructions
3. Exploring the latent space
"""

import numpy as np
import numpy_dl as ndl
from numpy_dl.models import Autoencoder
from numpy_dl.loss import MSELoss
from numpy_dl.optim import Adam
from numpy_dl.data import TensorDataset, DataLoader
from numpy_dl.utils import MetricTracker, accuracy


def generate_mnist_data(n_samples=1000):
    """Generate synthetic MNIST-like data for demonstration."""
    np.random.seed(42)
    # Generate random 28x28 images and flatten
    X = np.random.rand(n_samples, 784).astype(np.float32)
    # Add some structure (simple patterns)
    for i in range(n_samples):
        # Add horizontal or vertical lines
        img = X[i].reshape(28, 28)
        if i % 2 == 0:
            img[i % 28, :] = 1.0  # Horizontal line
        else:
            img[:, i % 28] = 1.0  # Vertical line
        X[i] = img.flatten()
    return X, X  # For autoencoders, input = target


def train_epoch(model, dataloader, optimizer, criterion, device='cpu'):
    """Train for one epoch."""
    model.train()
    metric_tracker = MetricTracker()

    for batch_idx, (data, target) in enumerate(dataloader):
        # Convert to tensors
        data = ndl.tensor(data, requires_grad=False, device=device)
        target = ndl.tensor(target, requires_grad=False, device=device)

        # Forward pass
        output = model(data)
        loss = criterion(output, target)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track metrics
        metric_tracker.update(loss=loss.item())

        if batch_idx % 10 == 0:
            print(f"  Batch {batch_idx}/{len(dataloader)}: Loss = {loss.item():.6f}")

    return metric_tracker.compute()


def evaluate(model, dataloader, criterion, device='cpu'):
    """Evaluate model."""
    model.eval()
    metric_tracker = MetricTracker()

    for data, target in dataloader:
        data = ndl.tensor(data, requires_grad=False, device=device)
        target = ndl.tensor(target, requires_grad=False, device=device)

        output = model(data)
        loss = criterion(output, target)

        metric_tracker.update(loss=loss.item())

    return metric_tracker.compute()


def visualize_reconstructions(model, test_data, n_samples=10, device='cpu'):
    """Visualize original vs reconstructed images."""
    model.eval()
    indices = np.random.choice(len(test_data), n_samples, replace=False)
    samples = test_data[indices]

    x = ndl.tensor(samples, requires_grad=False, device=device)
    recon = model(x)

    # Print reconstruction error for each sample
    print("\nReconstruction Errors:")
    for i in range(n_samples):
        error = np.mean((samples[i] - recon.data[i]) ** 2)
        print(f"  Sample {i+1}: MSE = {error:.6f}")


def explore_latent_space(model, test_data, device='cpu'):
    """Explore the latent space representation."""
    model.eval()
    x = ndl.tensor(test_data[:100], requires_grad=False, device=device)
    z = model.encode(x)

    print("\nLatent Space Statistics:")
    print(f"  Shape: {z.shape}")
    print(f"  Mean: {np.mean(z.data):.4f}")
    print(f"  Std: {np.std(z.data):.4f}")
    print(f"  Min: {np.min(z.data):.4f}")
    print(f"  Max: {np.max(z.data):.4f}")


def main():
    print("=" * 80)
    print("Basic Autoencoder for MNIST Reconstruction")
    print("=" * 80)

    # Hyperparameters
    input_size = 784  # 28x28 flattened
    hidden_sizes = [512, 256]
    latent_dim = 64
    batch_size = 128
    num_epochs = 10
    learning_rate = 0.001
    device = 'cpu'

    print("\nHyperparameters:")
    print(f"  Input size: {input_size}")
    print(f"  Hidden sizes: {hidden_sizes}")
    print(f"  Latent dimension: {latent_dim}")
    print(f"  Batch size: {batch_size}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Learning rate: {learning_rate}")

    # Generate data
    print("\nGenerating synthetic MNIST-like data...")
    X_train, _ = generate_mnist_data(n_samples=1000)
    X_test, _ = generate_mnist_data(n_samples=200)
    print(f"  Training samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")

    # Create datasets and dataloaders
    train_dataset = TensorDataset(X_train, X_train)
    test_dataset = TensorDataset(X_test, X_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Create model
    print("\nCreating autoencoder model...")
    model = Autoencoder(
        input_size=input_size,
        hidden_sizes=hidden_sizes,
        latent_dim=latent_dim,
        activation='relu'
    )
    print(f"  Model: {model}")
    print(f"  Total parameters: {sum(p.data.size for p in model.parameters())}")

    # Setup optimizer and loss
    optimizer = Adam(model.parameters(), lr=learning_rate)
    criterion = MSELoss()

    # Training loop
    print("\n" + "=" * 80)
    print("Training")
    print("=" * 80)

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 40)

        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device)
        test_metrics = evaluate(model, test_loader, criterion, device)

        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Train Loss: {train_metrics['loss']:.6f}")
        print(f"  Test Loss:  {test_metrics['loss']:.6f}")

    # Visualize results
    print("\n" + "=" * 80)
    print("Evaluation")
    print("=" * 80)

    visualize_reconstructions(model, X_test, n_samples=10, device=device)
    explore_latent_space(model, X_test, device=device)

    print("\n" + "=" * 80)
    print("Training completed successfully!")
    print("=" * 80)


if __name__ == '__main__':
    main()
