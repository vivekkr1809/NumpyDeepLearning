"""
Convolutional Variational Autoencoder for Image Generation.

This example demonstrates:
1. Training a convolutional VAE on image data
2. Leveraging spatial structure with convolutional layers
3. Generating high-quality image samples
4. Comparing with fully-connected VAE
"""

import numpy as np
import numpy_dl as ndl
from numpy_dl.models import ConvVariationalAutoencoder
from numpy_dl.loss import VAELoss
from numpy_dl.optim import Adam
from numpy_dl.data import TensorDataset, DataLoader
from numpy_dl.utils import MetricTracker


def generate_image_data(n_samples=1000, image_size=28):
    """Generate synthetic image data with spatial structure."""
    np.random.seed(42)
    images = np.zeros((n_samples, 1, image_size, image_size), dtype=np.float32)

    for i in range(n_samples):
        img = images[i, 0]

        # Create different geometric patterns
        pattern = i % 5

        if pattern == 0:
            # Horizontal stripes
            for j in range(0, image_size, 4):
                img[j:j+2, :] = 0.8

        elif pattern == 1:
            # Vertical stripes
            for j in range(0, image_size, 4):
                img[:, j:j+2] = 0.8

        elif pattern == 2:
            # Checkerboard
            for y in range(0, image_size, 4):
                for x in range(0, image_size, 4):
                    if (y + x) % 8 == 0:
                        img[y:y+4, x:x+4] = 0.8

        elif pattern == 3:
            # Circle
            center = image_size // 2
            radius = image_size // 3
            y, x = np.ogrid[:image_size, :image_size]
            mask = (x - center)**2 + (y - center)**2 <= radius**2
            img[mask] = 0.8

        else:
            # Square in center
            margin = image_size // 4
            img[margin:-margin, margin:-margin] = 0.8

        # Add noise
        img += np.random.randn(image_size, image_size).astype(np.float32) * 0.05
        img = np.clip(img, 0, 1)

    return images


def train_epoch(model, dataloader, optimizer, criterion, device='cpu'):
    """Train for one epoch."""
    model.train()
    metric_tracker = MetricTracker()

    for batch_idx, (data, _) in enumerate(dataloader):
        data = ndl.tensor(data, requires_grad=False, device=device)

        # Forward pass
        recon, mu, logvar = model(data)
        loss = criterion(recon, data, mu, logvar)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        metric_tracker.update(loss=loss.item())

        if batch_idx % 5 == 0:
            print(f"  Batch {batch_idx}/{len(dataloader)}: Loss = {loss.item():.6f}")

    return metric_tracker.compute()


def evaluate(model, dataloader, criterion, device='cpu'):
    """Evaluate model."""
    model.eval()
    metric_tracker = MetricTracker()

    for data, _ in dataloader:
        data = ndl.tensor(data, requires_grad=False, device=device)
        recon, mu, logvar = model(data)
        loss = criterion(recon, data, mu, logvar)
        metric_tracker.update(loss=loss.item())

    return metric_tracker.compute()


def generate_and_analyze_samples(model, num_samples=16, device='cpu'):
    """Generate samples and analyze quality."""
    print(f"\nGenerating {num_samples} samples...")
    model.eval()

    samples = model.sample(num_samples=num_samples, device=device)

    print(f"  Generated shape: {samples.shape}")
    print(f"  Statistics:")
    print(f"    Mean: {np.mean(samples.data):.4f}")
    print(f"    Std:  {np.std(samples.data):.4f}")
    print(f"    Min:  {np.min(samples.data):.4f}")
    print(f"    Max:  {np.max(samples.data):.4f}")

    return samples.data


def test_reconstruction_quality(model, test_data, device='cpu'):
    """Test reconstruction quality on test data."""
    print("\nTesting reconstruction quality...")
    model.eval()

    # Take first 10 samples
    x = ndl.tensor(test_data[:10], requires_grad=False, device=device)
    recon, _, _ = model(x)

    # Compute reconstruction errors
    errors = []
    for i in range(10):
        mse = np.mean((x.data[i] - recon.data[i]) ** 2)
        errors.append(mse)

    print(f"  Average MSE: {np.mean(errors):.6f}")
    print(f"  MSE std:     {np.std(errors):.6f}")
    print(f"  Min MSE:     {np.min(errors):.6f}")
    print(f"  Max MSE:     {np.max(errors):.6f}")


def explore_latent_interpolation(model, test_data, device='cpu'):
    """Explore latent space interpolation."""
    print("\nLatent space interpolation...")
    model.eval()

    # Select two images
    idx1, idx2 = 0, 1
    x1 = ndl.tensor(test_data[idx1:idx1+1], requires_grad=False, device=device)
    x2 = ndl.tensor(test_data[idx2:idx2+1], requires_grad=False, device=device)

    # Encode
    mu1, _ = model.encode(x1)
    mu2, _ = model.encode(x2)

    # Interpolate (5 steps)
    num_steps = 5
    print(f"  Interpolating with {num_steps} steps")

    for i, alpha in enumerate(np.linspace(0, 1, num_steps)):
        z = mu1.data * (1 - alpha) + mu2.data * alpha
        z_tensor = ndl.tensor(z, requires_grad=False, device=device)
        recon = model.decode(z_tensor)
        print(f"    Step {i+1}: α={alpha:.2f}, output shape={recon.shape}")


def main():
    print("=" * 80)
    print("Convolutional VAE for Image Generation")
    print("=" * 80)

    # Hyperparameters
    in_channels = 1
    base_channels = 32
    latent_dim = 64
    image_size = 28
    batch_size = 64
    num_epochs = 20
    learning_rate = 0.001
    beta = 1.0
    device = 'cpu'

    print("\nHyperparameters:")
    print(f"  Input channels: {in_channels}")
    print(f"  Base channels: {base_channels}")
    print(f"  Latent dimension: {latent_dim}")
    print(f"  Image size: {image_size}x{image_size}")
    print(f"  Batch size: {batch_size}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Beta: {beta}")

    # Generate data
    print("\nGenerating synthetic image data...")
    X_train = generate_image_data(n_samples=1000, image_size=image_size)
    X_test = generate_image_data(n_samples=200, image_size=image_size)
    print(f"  Training samples: {X_train.shape}")
    print(f"  Test samples: {X_test.shape}")

    # Create datasets
    train_dataset = TensorDataset(X_train, X_train)
    test_dataset = TensorDataset(X_test, X_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Create model
    print("\nCreating Convolutional VAE...")
    conv_vae = ConvVariationalAutoencoder(
        in_channels=in_channels,
        base_channels=base_channels,
        latent_dim=latent_dim,
        image_size=image_size
    )
    print(f"  Model: {conv_vae}")
    total_params = sum(p.data.size for p in conv_vae.parameters())
    print(f"  Total parameters: {total_params:,}")

    # Setup training
    optimizer = Adam(conv_vae.parameters(), lr=learning_rate)
    criterion = VAELoss(reconstruction_loss='mse', beta=beta, reduction='mean')

    # Training loop
    print("\n" + "=" * 80)
    print("Training")
    print("=" * 80)

    best_test_loss = float('inf')

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 40)

        train_metrics = train_epoch(conv_vae, train_loader, optimizer, criterion, device)
        test_metrics = evaluate(conv_vae, test_loader, criterion, device)

        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Train Loss: {train_metrics['loss']:.6f}")
        print(f"  Test Loss:  {test_metrics['loss']:.6f}")

        if test_metrics['loss'] < best_test_loss:
            best_test_loss = test_metrics['loss']
            print(f"  ★ New best test loss!")

    # Evaluation
    print("\n" + "=" * 80)
    print("Evaluation and Generation")
    print("=" * 80)

    test_reconstruction_quality(conv_vae, X_test, device=device)
    samples = generate_and_analyze_samples(conv_vae, num_samples=16, device=device)
    explore_latent_interpolation(conv_vae, X_test, device=device)

    print("\n" + "=" * 80)
    print("Training completed successfully!")
    print("=" * 80)
    print("\nAdvantages of Convolutional VAE:")
    print("  - Preserves spatial structure of images")
    print("  - More parameter-efficient than fully-connected")
    print("  - Better feature learning through convolutions")
    print("  - Smoother interpolations in latent space")
    print("  - Generates higher quality images")


if __name__ == '__main__':
    main()
