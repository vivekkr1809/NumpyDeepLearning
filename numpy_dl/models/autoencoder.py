"""Autoencoder architectures."""

import numpy as np
from numpy_dl.core.module import Module, Sequential
from numpy_dl.core.tensor import Tensor
from numpy_dl.nn import Linear, Conv2d, ConvTranspose2d, ReLU, Sigmoid, Tanh
from typing import List, Tuple, Optional


def get_array_module(data):
    """Get the appropriate array module (numpy or cupy) for the data."""
    import numpy as np
    if isinstance(data, np.ndarray):
        return np
    # Check for CuPy arrays
    try:
        import cupy as cp
        if isinstance(data, cp.ndarray):
            return cp
    except ImportError:
        pass
    return np


class Autoencoder(Module):
    """
    Standard fully-connected autoencoder.

    An autoencoder learns to compress data into a lower-dimensional latent space
    and then reconstruct it back to the original dimensions. It consists of an
    encoder that maps input to latent space and a decoder that reconstructs from latent space.

    Args:
        input_size: Dimension of input data
        hidden_sizes: List of hidden layer sizes for encoder (decoder is symmetric)
        latent_dim: Dimension of the latent space (bottleneck)
        activation: Activation function ('relu', 'sigmoid', 'tanh')
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int],
        latent_dim: int,
        activation: str = 'relu'
    ):
        super().__init__()
        self.input_size = input_size
        self.latent_dim = latent_dim
        self.hidden_sizes = hidden_sizes

        # Choose activation function
        if activation == 'relu':
            act_fn = ReLU
        elif activation == 'sigmoid':
            act_fn = Sigmoid
        elif activation == 'tanh':
            act_fn = Tanh
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Build encoder
        encoder_layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            encoder_layers.append(Linear(prev_size, hidden_size))
            encoder_layers.append(act_fn())
            prev_size = hidden_size
        encoder_layers.append(Linear(prev_size, latent_dim))
        self.encoder = Sequential(*encoder_layers)

        # Build decoder (symmetric to encoder)
        decoder_layers = []
        prev_size = latent_dim
        for hidden_size in reversed(hidden_sizes):
            decoder_layers.append(Linear(prev_size, hidden_size))
            decoder_layers.append(act_fn())
            prev_size = hidden_size
        decoder_layers.append(Linear(prev_size, input_size))
        # Final sigmoid to match data range [0, 1]
        decoder_layers.append(Sigmoid())
        self.decoder = Sequential(*decoder_layers)

    def encode(self, x: Tensor) -> Tensor:
        """
        Encode input to latent representation.

        Args:
            x: Input tensor of shape (batch_size, input_size)

        Returns:
            Latent representation of shape (batch_size, latent_dim)
        """
        return self.encoder(x)

    def decode(self, z: Tensor) -> Tensor:
        """
        Decode latent representation to reconstruction.

        Args:
            z: Latent tensor of shape (batch_size, latent_dim)

        Returns:
            Reconstructed output of shape (batch_size, input_size)
        """
        return self.decoder(z)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through encoder and decoder.

        Args:
            x: Input tensor of shape (batch_size, input_size)

        Returns:
            Reconstructed output of shape (batch_size, input_size)
        """
        z = self.encode(x)
        return self.decode(z)

    def __repr__(self):
        return (
            f"Autoencoder(input_size={self.input_size}, "
            f"hidden_sizes={self.hidden_sizes}, latent_dim={self.latent_dim})"
        )


class ConvAutoencoder(Module):
    """
    Convolutional autoencoder for image data.

    Uses convolutional layers in the encoder and transposed convolutions
    in the decoder for spatial data like images.

    Args:
        in_channels: Number of input channels (e.g., 1 for grayscale, 3 for RGB)
        base_channels: Base number of channels (doubled at each encoder layer)
        latent_dim: Dimension of the latent space
        image_size: Size of input images (assumes square images)
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 32,
        latent_dim: int = 128,
        image_size: int = 28
    ):
        super().__init__()
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.latent_dim = latent_dim
        self.image_size = image_size

        # Encoder: 28x28 -> 14x14 -> 7x7 -> latent
        self.encoder = Sequential(
            Conv2d(in_channels, base_channels, kernel_size=3, stride=2, padding=1),  # -> 14x14
            ReLU(),
            Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1),  # -> 7x7
            ReLU(),
            Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, stride=2, padding=1),  # -> 4x4
            ReLU(),
        )

        # Calculate the flattened size after convolutions
        # For 28x28 input with stride 2, padding 1, kernel 3:
        # output_size = (input + 2*pad - kernel) // stride + 1
        # 28 -> (28+2-3)//2+1 = 14
        # 14 -> (14+2-3)//2+1 = 7
        # 7  -> (7+2-3)//2+1  = 4
        h = image_size
        for _ in range(3):  # 3 conv layers
            h = (h + 2 * 1 - 3) // 2 + 1
        self.conv_output_size = h * h * (base_channels * 4)
        self.decoder_input_size = h

        # Latent space
        self.fc_encode = Linear(self.conv_output_size, latent_dim)
        self.fc_decode = Linear(latent_dim, self.conv_output_size)

        # Decoder: latent -> 4x4 -> 8x8 -> 16x16 -> 32x32
        # Note: We'll crop to match input size if needed
        self.decoder = Sequential(
            ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=4, stride=2, padding=1),  # -> 8x8
            ReLU(),
            ConvTranspose2d(base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1),  # -> 16x16
            ReLU(),
            ConvTranspose2d(base_channels, in_channels, kernel_size=4, stride=2, padding=1),  # -> 32x32
            Sigmoid()
        )

    def encode(self, x: Tensor) -> Tensor:
        """
        Encode input image to latent representation.

        Args:
            x: Input tensor of shape (batch_size, in_channels, height, width)

        Returns:
            Latent representation of shape (batch_size, latent_dim)
        """
        batch_size = x.shape[0]
        x = self.encoder(x)
        x = x.reshape(batch_size, -1)
        return self.fc_encode(x)

    def decode(self, z: Tensor) -> Tensor:
        """
        Decode latent representation to image.

        Args:
            z: Latent tensor of shape (batch_size, latent_dim)

        Returns:
            Reconstructed image of shape (batch_size, in_channels, height, width)
        """
        batch_size = z.shape[0]
        x = self.fc_decode(z)
        x = x.reshape(batch_size, self.base_channels * 4, self.decoder_input_size, self.decoder_input_size)
        x = self.decoder(x)

        # Crop to match input size if necessary
        if x.shape[2] != self.image_size or x.shape[3] != self.image_size:
            # Center crop
            h_start = (x.shape[2] - self.image_size) // 2
            w_start = (x.shape[3] - self.image_size) // 2
            x_data = x.data[:, :, h_start:h_start + self.image_size, w_start:w_start + self.image_size]
            x = Tensor(
                x_data,
                requires_grad=x.requires_grad,
                device=x.device,
                _children=(x,),
                _op='crop'
            )

            def _backward():
                if x.requires_grad:
                    # Create gradient tensor with same shape as input
                    xp = get_array_module(x.data)
                    grad_full = xp.zeros(x._children[0].shape, dtype=x.dtype)
                    # Copy gradients to cropped region
                    grad_full[:, :, h_start:h_start + self.image_size, w_start:w_start + self.image_size] = x.grad
                    x._children[0].grad = grad_full if x._children[0].grad is None else x._children[0].grad + grad_full

            x._backward = _backward

        return x

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through encoder and decoder.

        Args:
            x: Input tensor of shape (batch_size, in_channels, height, width)

        Returns:
            Reconstructed output of same shape as input
        """
        z = self.encode(x)
        return self.decode(z)

    def __repr__(self):
        return (
            f"ConvAutoencoder(in_channels={self.in_channels}, "
            f"base_channels={self.base_channels}, latent_dim={self.latent_dim}, "
            f"image_size={self.image_size})"
        )


class VariationalAutoencoder(Module):
    """
    Variational Autoencoder (VAE).

    A VAE learns a probabilistic mapping from input to latent space by encoding
    inputs as distributions rather than fixed points. This enables sampling and
    generation of new data.

    The encoder outputs mean (mu) and log-variance (logvar) of the latent distribution,
    and samples are drawn using the reparameterization trick: z = mu + sigma * epsilon.

    Args:
        input_size: Dimension of input data
        hidden_sizes: List of hidden layer sizes for encoder (decoder is symmetric)
        latent_dim: Dimension of the latent space
        activation: Activation function ('relu', 'sigmoid', 'tanh')
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int],
        latent_dim: int,
        activation: str = 'relu'
    ):
        super().__init__()
        self.input_size = input_size
        self.latent_dim = latent_dim
        self.hidden_sizes = hidden_sizes

        # Choose activation function
        if activation == 'relu':
            act_fn = ReLU
        elif activation == 'sigmoid':
            act_fn = Sigmoid
        elif activation == 'tanh':
            act_fn = Tanh
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Build encoder backbone
        encoder_layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            encoder_layers.append(Linear(prev_size, hidden_size))
            encoder_layers.append(act_fn())
            prev_size = hidden_size
        self.encoder_backbone = Sequential(*encoder_layers)

        # Latent distribution parameters
        self.fc_mu = Linear(prev_size, latent_dim)
        self.fc_logvar = Linear(prev_size, latent_dim)

        # Build decoder (symmetric to encoder)
        decoder_layers = []
        prev_size = latent_dim
        for hidden_size in reversed(hidden_sizes):
            decoder_layers.append(Linear(prev_size, hidden_size))
            decoder_layers.append(act_fn())
            prev_size = hidden_size
        decoder_layers.append(Linear(prev_size, input_size))
        decoder_layers.append(Sigmoid())
        self.decoder = Sequential(*decoder_layers)

    def encode(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Encode input to latent distribution parameters.

        Args:
            x: Input tensor of shape (batch_size, input_size)

        Returns:
            Tuple of (mu, logvar), each of shape (batch_size, latent_dim)
            - mu: Mean of the latent distribution
            - logvar: Log-variance of the latent distribution
        """
        h = self.encoder_backbone(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick: z = mu + sigma * epsilon.

        Args:
            mu: Mean tensor of shape (batch_size, latent_dim)
            logvar: Log-variance tensor of shape (batch_size, latent_dim)

        Returns:
            Sampled latent vector of shape (batch_size, latent_dim)
        """
        xp = get_array_module(mu.data)
        # std = exp(0.5 * logvar)
        std_data = xp.exp(0.5 * logvar.data)
        # Sample epsilon from standard normal
        eps = xp.random.randn(*mu.shape).astype(mu.dtype)
        # z = mu + std * eps
        z_data = mu.data + std_data * eps

        z = Tensor(
            z_data,
            requires_grad=mu.requires_grad or logvar.requires_grad,
            device=mu.device,
            _children=(mu, logvar),
            _op='reparameterize'
        )

        def _backward():
            if mu.requires_grad:
                # Gradient w.r.t. mu is just the output gradient
                mu.grad = z.grad if mu.grad is None else mu.grad + z.grad

            if logvar.requires_grad:
                # Gradient w.r.t. logvar: d(mu + exp(0.5*logvar)*eps)/d(logvar)
                #                       = 0.5 * exp(0.5*logvar) * eps
                grad_logvar = 0.5 * std_data * eps * z.grad
                logvar.grad = grad_logvar if logvar.grad is None else logvar.grad + grad_logvar

        z._backward = _backward
        return z

    def decode(self, z: Tensor) -> Tensor:
        """
        Decode latent representation to reconstruction.

        Args:
            z: Latent tensor of shape (batch_size, latent_dim)

        Returns:
            Reconstructed output of shape (batch_size, input_size)
        """
        return self.decoder(z)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Forward pass through VAE.

        Args:
            x: Input tensor of shape (batch_size, input_size)

        Returns:
            Tuple of (reconstruction, mu, logvar)
            - reconstruction: Reconstructed output of shape (batch_size, input_size)
            - mu: Mean of latent distribution of shape (batch_size, latent_dim)
            - logvar: Log-variance of latent distribution of shape (batch_size, latent_dim)
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        return reconstruction, mu, logvar

    def sample(self, num_samples: int, device: str = 'cpu') -> Tensor:
        """
        Generate new samples by sampling from the latent space.

        Args:
            num_samples: Number of samples to generate
            device: Device to create samples on ('cpu' or 'cuda')

        Returns:
            Generated samples of shape (num_samples, input_size)
        """
        xp = np if device == 'cpu' else get_array_module(np.array([]))
        # Sample from standard normal distribution
        z_data = xp.random.randn(num_samples, self.latent_dim).astype(np.float32)
        z = Tensor(z_data, requires_grad=False, device=device)
        return self.decode(z)

    def __repr__(self):
        return (
            f"VariationalAutoencoder(input_size={self.input_size}, "
            f"hidden_sizes={self.hidden_sizes}, latent_dim={self.latent_dim})"
        )


class ConvVariationalAutoencoder(Module):
    """
    Convolutional Variational Autoencoder (Conv-VAE).

    Combines convolutional architecture with variational inference for
    learning generative models of images.

    Args:
        in_channels: Number of input channels (e.g., 1 for grayscale, 3 for RGB)
        base_channels: Base number of channels (doubled at each encoder layer)
        latent_dim: Dimension of the latent space
        image_size: Size of input images (assumes square images)
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 32,
        latent_dim: int = 128,
        image_size: int = 28
    ):
        super().__init__()
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.latent_dim = latent_dim
        self.image_size = image_size

        # Encoder
        self.encoder = Sequential(
            Conv2d(in_channels, base_channels, kernel_size=3, stride=2, padding=1),
            ReLU(),
            Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1),
            ReLU(),
            Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, stride=2, padding=1),
            ReLU(),
        )

        # Calculate the flattened size after convolutions
        h = image_size
        for _ in range(3):  # 3 conv layers
            h = (h + 2 * 1 - 3) // 2 + 1
        self.conv_output_size = h * h * (base_channels * 4)
        self.decoder_input_size = h

        # Latent distribution parameters
        self.fc_mu = Linear(self.conv_output_size, latent_dim)
        self.fc_logvar = Linear(self.conv_output_size, latent_dim)

        # Decoder input
        self.fc_decode = Linear(latent_dim, self.conv_output_size)

        # Decoder
        self.decoder = Sequential(
            ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=4, stride=2, padding=1),
            ReLU(),
            ConvTranspose2d(base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1),
            ReLU(),
            ConvTranspose2d(base_channels, in_channels, kernel_size=4, stride=2, padding=1),
            Sigmoid()
        )

    def encode(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Encode input image to latent distribution parameters.

        Args:
            x: Input tensor of shape (batch_size, in_channels, height, width)

        Returns:
            Tuple of (mu, logvar), each of shape (batch_size, latent_dim)
        """
        batch_size = x.shape[0]
        h = self.encoder(x)
        h = h.reshape(batch_size, -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick: z = mu + sigma * epsilon.

        Args:
            mu: Mean tensor of shape (batch_size, latent_dim)
            logvar: Log-variance tensor of shape (batch_size, latent_dim)

        Returns:
            Sampled latent vector of shape (batch_size, latent_dim)
        """
        xp = get_array_module(mu.data)
        std_data = xp.exp(0.5 * logvar.data)
        eps = xp.random.randn(*mu.shape).astype(mu.dtype)
        z_data = mu.data + std_data * eps

        z = Tensor(
            z_data,
            requires_grad=mu.requires_grad or logvar.requires_grad,
            device=mu.device,
            _children=(mu, logvar),
            _op='reparameterize'
        )

        def _backward():
            if mu.requires_grad:
                mu.grad = z.grad if mu.grad is None else mu.grad + z.grad

            if logvar.requires_grad:
                grad_logvar = 0.5 * std_data * eps * z.grad
                logvar.grad = grad_logvar if logvar.grad is None else logvar.grad + grad_logvar

        z._backward = _backward
        return z

    def decode(self, z: Tensor) -> Tensor:
        """
        Decode latent representation to image.

        Args:
            z: Latent tensor of shape (batch_size, latent_dim)

        Returns:
            Reconstructed image of shape (batch_size, in_channels, height, width)
        """
        batch_size = z.shape[0]
        h = self.fc_decode(z)
        h = h.reshape(batch_size, self.base_channels * 4, self.decoder_input_size, self.decoder_input_size)
        x = self.decoder(h)

        # Crop to match input size if necessary
        if x.shape[2] != self.image_size or x.shape[3] != self.image_size:
            # Center crop
            h_start = (x.shape[2] - self.image_size) // 2
            w_start = (x.shape[3] - self.image_size) // 2
            x_data = x.data[:, :, h_start:h_start + self.image_size, w_start:w_start + self.image_size]
            x = Tensor(
                x_data,
                requires_grad=x.requires_grad,
                device=x.device,
                _children=(x,),
                _op='crop'
            )

            def _backward():
                if x.requires_grad:
                    xp = get_array_module(x.data)
                    grad_full = xp.zeros(x._children[0].shape, dtype=x.dtype)
                    grad_full[:, :, h_start:h_start + self.image_size, w_start:w_start + self.image_size] = x.grad
                    x._children[0].grad = grad_full if x._children[0].grad is None else x._children[0].grad + grad_full

            x._backward = _backward

        return x

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Forward pass through Conv-VAE.

        Args:
            x: Input tensor of shape (batch_size, in_channels, height, width)

        Returns:
            Tuple of (reconstruction, mu, logvar)
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        return reconstruction, mu, logvar

    def sample(self, num_samples: int, device: str = 'cpu') -> Tensor:
        """
        Generate new samples by sampling from the latent space.

        Args:
            num_samples: Number of samples to generate
            device: Device to create samples on ('cpu' or 'cuda')

        Returns:
            Generated samples of shape (num_samples, in_channels, height, width)
        """
        xp = np if device == 'cpu' else get_array_module(np.array([]))
        z_data = xp.random.randn(num_samples, self.latent_dim).astype(np.float32)
        z = Tensor(z_data, requires_grad=False, device=device)
        return self.decode(z)

    def __repr__(self):
        return (
            f"ConvVariationalAutoencoder(in_channels={self.in_channels}, "
            f"base_channels={self.base_channels}, latent_dim={self.latent_dim}, "
            f"image_size={self.image_size})"
        )
