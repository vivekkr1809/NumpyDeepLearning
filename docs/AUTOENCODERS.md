# Autoencoders

This document provides a comprehensive guide to using autoencoders in the NumpyDeepLearning framework.

## Table of Contents

1. [Overview](#overview)
2. [Architectures](#architectures)
3. [API Reference](#api-reference)
4. [Usage Examples](#usage-examples)
5. [Training Tips](#training-tips)
6. [Applications](#applications)

## Overview

Autoencoders are neural networks designed to learn efficient data representations in an unsupervised manner. They consist of two main components:

- **Encoder**: Compresses input data into a lower-dimensional latent representation
- **Decoder**: Reconstructs the original input from the latent representation

### Why Use Autoencoders?

Autoencoders are useful for various tasks:

1. **Dimensionality Reduction**: Learn compact representations of high-dimensional data
2. **Denoising**: Remove noise from corrupted inputs
3. **Anomaly Detection**: Identify unusual patterns in data
4. **Feature Learning**: Learn useful features for downstream tasks
5. **Data Generation** (VAE): Generate new samples similar to training data

## Architectures

The framework provides four autoencoder variants:

### 1. Standard Autoencoder

A fully-connected autoencoder for vector data (e.g., flattened images, tabular data).

**Architecture**:
```
Input → [FC → Activation]* → Latent → [FC → Activation]* → Output
```

**Best for**: Tabular data, flattened images, feature vectors

### 2. Convolutional Autoencoder

Uses convolutional layers for spatial data like images. More parameter-efficient for visual data.

**Architecture**:
```
Input → [Conv → ReLU]* → Flatten → FC → Latent
Latent → FC → Reshape → [ConvTranspose → ReLU]* → Output
```

**Best for**: Images, spatial data, feature maps

### 3. Variational Autoencoder (VAE)

A probabilistic autoencoder that learns a distribution over the latent space, enabling generation of new samples.

**Architecture**:
```
Input → Encoder → (μ, log σ²)
z = μ + σ * ε (where ε ~ N(0,1))
z → Decoder → Output
```

**Loss**: Reconstruction loss + KL divergence

**Best for**: Generative modeling, sampling, interpolation

### 4. Convolutional VAE

Combines convolutional architecture with variational inference for image generation.

**Best for**: Image generation, style transfer, image interpolation

## API Reference

### Autoencoder

```python
from numpy_dl.models import Autoencoder

model = Autoencoder(
    input_size=784,          # Dimension of input vectors
    hidden_sizes=[512, 256], # Hidden layer sizes for encoder
    latent_dim=64,           # Dimension of latent space
    activation='relu'        # Activation function
)
```

**Methods**:
- `encode(x)`: Encode input to latent representation
- `decode(z)`: Decode latent vector to reconstruction
- `forward(x)`: Full encoding and decoding

### ConvAutoencoder

```python
from numpy_dl.models import ConvAutoencoder

model = ConvAutoencoder(
    in_channels=1,       # Number of input channels (1 for grayscale, 3 for RGB)
    base_channels=32,    # Base number of channels (doubled in each layer)
    latent_dim=128,      # Dimension of latent space
    image_size=28        # Input image size (assumes square images)
)
```

### VariationalAutoencoder

```python
from numpy_dl.models import VariationalAutoencoder

model = VariationalAutoencoder(
    input_size=784,
    hidden_sizes=[512, 256],
    latent_dim=64,
    activation='relu'
)
```

**Additional Methods**:
- `encode(x)`: Returns `(mu, logvar)` - mean and log-variance of latent distribution
- `reparameterize(mu, logvar)`: Sample from latent distribution using reparameterization trick
- `sample(num_samples, device)`: Generate new samples from the learned distribution

### ConvVariationalAutoencoder

```python
from numpy_dl.models import ConvVariationalAutoencoder

model = ConvVariationalAutoencoder(
    in_channels=1,
    base_channels=32,
    latent_dim=128,
    image_size=28
)
```

### Loss Functions

#### VAELoss

Combined reconstruction and KL divergence loss for VAE training:

```python
from numpy_dl.loss import VAELoss

criterion = VAELoss(
    reconstruction_loss='mse',  # 'mse' or 'bce'
    beta=1.0,                   # Weight for KL term (beta-VAE)
    reduction='mean'            # 'mean', 'sum', or 'none'
)

# Usage
recon, mu, logvar = vae(x)
loss = criterion(recon, x, mu, logvar)
```

#### KLDivergenceLoss

Standalone KL divergence loss:

```python
from numpy_dl.loss import KLDivergenceLoss

kl_loss = KLDivergenceLoss(reduction='mean')
mu, logvar = vae.encode(x)
kl = kl_loss(mu, logvar)
```

## Usage Examples

### Training a Standard Autoencoder

```python
import numpy as np
import numpy_dl as ndl
from numpy_dl.models import Autoencoder
from numpy_dl.loss import MSELoss
from numpy_dl.optim import Adam
from numpy_dl.data import TensorDataset, DataLoader

# Create model
model = Autoencoder(
    input_size=784,
    hidden_sizes=[512, 256],
    latent_dim=64
)

# Setup training
optimizer = Adam(model.parameters(), lr=0.001)
criterion = MSELoss()

# Create dataloader
dataset = TensorDataset(X_train, X_train)  # Input = target for autoencoders
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

# Training loop
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0

    for batch_x, _ in dataloader:
        # Convert to tensors
        x = ndl.tensor(batch_x, requires_grad=False)

        # Forward pass
        recon = model(x)
        loss = criterion(recon, x)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch {epoch+1}: Loss = {epoch_loss / len(dataloader):.4f}")
```

### Training a Variational Autoencoder

```python
from numpy_dl.models import VariationalAutoencoder
from numpy_dl.loss import VAELoss

# Create VAE
vae = VariationalAutoencoder(
    input_size=784,
    hidden_sizes=[512, 256],
    latent_dim=64
)

# VAE-specific loss
criterion = VAELoss(
    reconstruction_loss='mse',
    beta=1.0,  # Standard VAE
    reduction='mean'
)

# Training loop
for epoch in range(num_epochs):
    vae.train()
    epoch_loss = 0.0

    for batch_x, _ in dataloader:
        x = ndl.tensor(batch_x, requires_grad=False)

        # VAE forward returns (reconstruction, mu, logvar)
        recon, mu, logvar = vae(x)

        # Compute VAE loss
        loss = criterion(recon, x, mu, logvar)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch {epoch+1}: Loss = {epoch_loss / len(dataloader):.4f}")
```

### Generating New Samples with VAE

```python
# After training, generate new samples
vae.eval()

# Generate 10 new samples
samples = vae.sample(num_samples=10, device='cpu')

# samples.shape = (10, 784) for standard VAE
# or (10, 1, 28, 28) for ConvVAE
```

### Latent Space Interpolation

```python
# Encode two images
vae.eval()
x1 = ndl.tensor(image1, requires_grad=False)
x2 = ndl.tensor(image2, requires_grad=False)

mu1, _ = vae.encode(x1)
mu2, _ = vae.encode(x2)

# Interpolate in latent space
num_steps = 10
alphas = np.linspace(0, 1, num_steps)
interpolations = []

for alpha in alphas:
    z = mu1 * (1 - alpha) + mu2 * alpha
    recon = vae.decode(z)
    interpolations.append(recon.data)
```

### Training a Convolutional Autoencoder

```python
from numpy_dl.models import ConvAutoencoder

# Create convolutional autoencoder
model = ConvAutoencoder(
    in_channels=1,       # Grayscale images
    base_channels=32,
    latent_dim=128,
    image_size=28
)

# Training is similar to standard autoencoder
# Input shape: (batch_size, channels, height, width)
for batch_x, _ in dataloader:
    # batch_x shape: (batch_size, 1, 28, 28)
    x = ndl.tensor(batch_x, requires_grad=False)

    recon = model(x)
    loss = criterion(recon, x)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

## Training Tips

### 1. Choosing Latent Dimension

- **Too small**: Underfitting, poor reconstruction
- **Too large**: Overfitting, less compression
- **Rule of thumb**: Start with 10-20% of input dimension
- **Example**: For 784-dimensional input, try 64-128

### 2. Architecture Design

**For Standard Autoencoders**:
- Use symmetric encoder/decoder
- Gradually reduce dimensions in encoder
- Mirror the structure in decoder

**For Convolutional Autoencoders**:
- Use stride=2 for downsampling in encoder
- Use ConvTranspose2d with stride=2 for upsampling in decoder
- Match spatial dimensions between encoder and decoder

### 3. Activation Functions

- **ReLU**: Default choice, works well for most cases
- **Sigmoid/Tanh**: Better for inputs in [0,1] or [-1,1]
- **Final layer**: Sigmoid for binary/normalized data, linear for unbounded data

### 4. Loss Functions

**For Standard Autoencoders**:
- **MSE**: Good for continuous data, easy to optimize
- **BCE**: Better for binary data (e.g., binarized images)

**For VAEs**:
- **Beta parameter**: Controls KL weight
  - β = 1: Standard VAE
  - β > 1: Disentangled representations (β-VAE)
  - β < 1: Better reconstruction, less regularization

### 5. Learning Rate

- Start with 0.001 for Adam optimizer
- Use learning rate decay for better convergence
- Monitor both reconstruction and total loss

### 6. Regularization

- Add L2 regularization to prevent overfitting
- Use dropout in encoder/decoder (not in latent space)
- For VAE, KL term acts as regularization

### 7. Data Preprocessing

- **Normalize** inputs to [0, 1] or [-1, 1]
- **Standardize** if using MSE loss
- **Augmentation** can improve robustness

## Applications

### 1. Dimensionality Reduction

```python
# Train autoencoder
model = Autoencoder(input_size=784, hidden_sizes=[256], latent_dim=32)
# ... train model ...

# Extract low-dimensional features
model.eval()
z = model.encode(ndl.tensor(X, requires_grad=False))
features = z.data  # Shape: (n_samples, 32)
```

### 2. Denoising

```python
# Train on clean data, but add noise during training
for batch_x, _ in dataloader:
    # Add noise
    noisy_x = batch_x + np.random.normal(0, 0.1, batch_x.shape)
    noisy_x = np.clip(noisy_x, 0, 1)

    # Train to reconstruct clean from noisy
    x_noisy = ndl.tensor(noisy_x, requires_grad=False)
    x_clean = ndl.tensor(batch_x, requires_grad=False)

    recon = model(x_noisy)
    loss = criterion(recon, x_clean)
```

### 3. Anomaly Detection

```python
# Train on normal data only
# Anomalies have high reconstruction error

model.eval()
x = ndl.tensor(test_sample, requires_grad=False)
recon = model(x)

# Compute reconstruction error
error = np.mean((x.data - recon.data) ** 2)
is_anomaly = error > threshold
```

### 4. Data Generation (VAE only)

```python
# Generate random samples
vae.eval()
new_samples = vae.sample(num_samples=100, device='cpu')

# Interpolate between samples
# (see Latent Space Interpolation example above)
```

### 5. Transfer Learning

```python
# Use encoder for feature extraction
model = Autoencoder(...)
# ... train on large dataset ...

# Extract encoder
encoder = model.encoder

# Use encoder features for downstream task
class Classifier(Module):
    def __init__(self, encoder, num_classes):
        super().__init__()
        self.encoder = encoder
        self.classifier = Linear(latent_dim, num_classes)

    def forward(self, x):
        # Freeze encoder if desired
        features = self.encoder(x)
        return self.classifier(features)
```

## Advanced Topics

### Beta-VAE for Disentanglement

Use β > 1 to encourage learning disentangled representations:

```python
criterion = VAELoss(
    reconstruction_loss='mse',
    beta=4.0,  # Stronger KL regularization
    reduction='mean'
)
```

### Conditional VAE

Extend VAE to condition on labels:

```python
class ConditionalVAE(VariationalAutoencoder):
    def __init__(self, input_size, num_classes, ...):
        super().__init__(input_size + num_classes, ...)
        self.num_classes = num_classes

    def forward(self, x, labels):
        # One-hot encode labels
        labels_onehot = to_one_hot(labels, self.num_classes)
        # Concatenate with input
        x_cond = ndl.cat([x, labels_onehot], dim=1)
        return super().forward(x_cond)
```

### Sparse Autoencoders

Add L1 regularization on latent activations:

```python
z = model.encode(x)
sparsity_loss = 0.001 * ndl.abs(z).mean()
total_loss = reconstruction_loss + sparsity_loss
```

## References

1. Hinton, G. E., & Salakhutdinov, R. R. (2006). "Reducing the dimensionality of data with neural networks"
2. Kingma, D. P., & Welling, M. (2013). "Auto-Encoding Variational Bayes"
3. Higgins, I., et al. (2017). "β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework"
4. Doersch, C. (2016). "Tutorial on Variational Autoencoders"

## See Also

- [Full API Documentation](../README.md)
- [Examples](../examples/autoencoders/)
- [Multi-Task Learning](MULTITASK_LEARNING.md)
