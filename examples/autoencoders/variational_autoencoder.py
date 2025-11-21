"""
Variational Autoencoder (VAE) with Sampling and Interpolation.

This example demonstrates:
1. Training a VAE on MNIST-like data
2. Generating new samples from the learned distribution
3. Interpolating between images in latent space
4. Exploring the structured latent space
"""

import numpy as np
import numpy_dl as ndl
from numpy_dl.models import VariationalAutoencoder
from numpy_dl.loss import VAELoss
from numpy_dl.optim import Adam
from numpy_dl.data import TensorDataset, DataLoader
from numpy_dl.utils import MetricTracker


def generate_mnist_data(n_samples=1000):
    """Generate synthetic MNIST-like data."""
    np.random.seed(42)
    X = np.random.rand(n_samples, 784).astype(np.float32)
    # Add structure
    for i in range(n_samples):
        img = X[i].reshape(28, 28)
        # Create simple patterns
        pattern = i % 4
        if pattern == 0:
            img[10:18, :] = 0.8  # Horizontal bar
        elif pattern == 1:
            img[:, 10:18] = 0.8  # Vertical bar
        elif pattern == 2:
            img[10:18, 10:18] = 0.8  # Square
        else:
            np.fill_diagonal(img, 0.8)  # Diagonal
        X[i] = img.flatten()
    return X


def train_epoch(model, dataloader, optimizer, criterion, device='cpu'):
    """Train VAE for one epoch."""
    model.train()
    metric_tracker = MetricTracker()

    for batch_idx, (data, _) in enumerate(dataloader):
        data = ndl.tensor(data, requires_grad=False, device=device)

        # VAE forward pass returns (reconstruction, mu, logvar)
        recon, mu, logvar = model(data)

        # Compute VAE loss (reconstruction + KL divergence)
        loss = criterion(recon, data, mu, logvar)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        metric_tracker.update(loss=loss.item())

        if batch_idx % 10 == 0:
            print(f"  Batch {batch_idx}/{len(dataloader)}: Loss = {loss.item():.6f}")

    return metric_tracker.compute()


def evaluate(model, dataloader, criterion, device='cpu'):
    """Evaluate VAE."""
    model.eval()
    metric_tracker = MetricTracker()

    for data, _ in dataloader:
        data = ndl.tensor(data, requires_grad=False, device=device)
        recon, mu, logvar = model(data)
        loss = criterion(recon, data, mu, logvar)
        metric_tracker.update(loss=loss.item())

    return metric_tracker.compute()


def generate_samples(model, num_samples=10, device='cpu'):
    """Generate new samples from the learned distribution."""
    print(f"\nGenerating {num_samples} new samples...")
    model.eval()

    samples = model.sample(num_samples=num_samples, device=device)

    print(f"  Generated samples shape: {samples.shape}")
    print(f"  Sample statistics:")
    print(f"    Mean: {np.mean(samples.data):.4f}")
    print(f"    Std:  {np.std(samples.data):.4f}")
    print(f"    Min:  {np.min(samples.data):.4f}")
    print(f"    Max:  {np.max(samples.data):.4f}")

    return samples.data


def interpolate_latent_space(model, data, num_steps=10, device='cpu'):
    """Interpolate between two images in latent space."""
    print(f"\nInterpolating between two images ({num_steps} steps)...")
    model.eval()

    # Select two random images
    idx1, idx2 = np.random.choice(len(data), 2, replace=False)
    x1 = ndl.tensor(data[idx1:idx1+1], requires_grad=False, device=device)
    x2 = ndl.tensor(data[idx2:idx2+1], requires_grad=False, device=device)

    # Encode to latent space
    mu1, _ = model.encode(x1)
    mu2, _ = model.encode(x2)

    print(f"  Image 1 latent vector (first 5 dims): {mu1.data[0, :5]}")
    print(f"  Image 2 latent vector (first 5 dims): {mu2.data[0, :5]}")

    # Interpolate
    interpolations = []
    alphas = np.linspace(0, 1, num_steps)

    for alpha in alphas:
        # Linear interpolation in latent space
        z_interp_data = mu1.data * (1 - alpha) + mu2.data * alpha
        z_interp = ndl.tensor(z_interp_data, requires_grad=False, device=device)

        # Decode
        recon = model.decode(z_interp)
        interpolations.append(recon.data[0])

    interpolations = np.array(interpolations)
    print(f"  Interpolation shape: {interpolations.shape}")

    return interpolations


def analyze_latent_space(model, data, device='cpu'):
    """Analyze the structure of the learned latent space."""
    print("\nAnalyzing latent space...")
    model.eval()

    x = ndl.tensor(data[:200], requires_grad=False, device=device)
    mu, logvar = model.encode(x)

    print(f"  Latent space dimension: {mu.shape[1]}")
    print(f"\n  Mean (μ) statistics:")
    print(f"    Mean: {np.mean(mu.data):.4f}")
    print(f"    Std:  {np.std(mu.data):.4f}")
    print(f"    Min:  {np.min(mu.data):.4f}")
    print(f"    Max:  {np.max(mu.data):.4f}")

    print(f"\n  Log-variance (log σ²) statistics:")
    print(f"    Mean: {np.mean(logvar.data):.4f}")
    print(f"    Std:  {np.std(logvar.data):.4f}")
    print(f"    Min:  {np.min(logvar.data):.4f}")
    print(f"    Max:  {np.max(logvar.data):.4f}")

    # Check if distribution is close to standard normal
    print(f"\n  Distance from standard normal N(0,1):")
    print(f"    Mean difference: {np.abs(np.mean(mu.data) - 0.0):.4f}")
    print(f"    Std difference:  {np.abs(np.std(mu.data) - 1.0):.4f}")


def main():
    print("=" * 80)
    print("Variational Autoencoder (VAE) Training and Generation")
    print("=" * 80)

    # Hyperparameters
    input_size = 784
    hidden_sizes = [512, 256]
    latent_dim = 32
    batch_size = 128
    num_epochs = 15
    learning_rate = 0.001
    beta = 1.0  # Standard VAE (beta-VAE uses beta > 1)
    device = 'cpu'

    print("\nHyperparameters:")
    print(f"  Input size: {input_size}")
    print(f"  Hidden sizes: {hidden_sizes}")
    print(f"  Latent dimension: {latent_dim}")
    print(f"  Batch size: {batch_size}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Beta (KL weight): {beta}")

    # Generate data
    print("\nGenerating synthetic data...")
    X_train = generate_mnist_data(n_samples=1200)
    X_test = generate_mnist_data(n_samples=200)
    print(f"  Training samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")

    # Create datasets and dataloaders
    train_dataset = TensorDataset(X_train, X_train)
    test_dataset = TensorDataset(X_test, X_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Create VAE model
    print("\nCreating VAE model...")
    vae = VariationalAutoencoder(
        input_size=input_size,
        hidden_sizes=hidden_sizes,
        latent_dim=latent_dim,
        activation='relu'
    )
    print(f"  Model: {vae}")
    print(f"  Total parameters: {sum(p.data.size for p in vae.parameters())}")

    # Setup optimizer and VAE loss
    optimizer = Adam(vae.parameters(), lr=learning_rate)
    criterion = VAELoss(
        reconstruction_loss='mse',
        beta=beta,
        reduction='mean'
    )
    print(f"\n  Loss: {criterion}")

    # Training loop
    print("\n" + "=" * 80)
    print("Training")
    print("=" * 80)

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 40)

        train_metrics = train_epoch(vae, train_loader, optimizer, criterion, device)
        test_metrics = evaluate(vae, test_loader, criterion, device)

        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Train Loss: {train_metrics['loss']:.6f}")
        print(f"  Test Loss:  {test_metrics['loss']:.6f}")

    # Evaluation and generation
    print("\n" + "=" * 80)
    print("Generation and Exploration")
    print("=" * 80)

    # 1. Analyze latent space
    analyze_latent_space(vae, X_test, device=device)

    # 2. Generate new samples
    samples = generate_samples(vae, num_samples=20, device=device)

    # 3. Interpolate in latent space
    interpolations = interpolate_latent_space(vae, X_test, num_steps=10, device=device)

    print("\n" + "=" * 80)
    print("Training and generation completed successfully!")
    print("=" * 80)
    print("\nKey Takeaways:")
    print("  - VAE learns a smooth, structured latent space")
    print("  - Can generate novel samples by sampling from N(0,1)")
    print("  - Interpolations in latent space produce meaningful transitions")
    print("  - KL divergence regularizes latent space to be close to N(0,1)")


if __name__ == '__main__':
    main()
