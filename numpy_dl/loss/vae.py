"""Loss functions for Variational Autoencoders."""

import numpy as np
from numpy_dl.core.module import Module
from numpy_dl.core.tensor import Tensor
from numpy_dl.utils.device import get_array_module
from typing import Optional


class VAELoss(Module):
    """
    Variational Autoencoder Loss.

    Combines reconstruction loss with KL divergence loss:
        Loss = reconstruction_loss + beta * KL_divergence

    The reconstruction loss measures how well the decoder reconstructs the input,
    and the KL divergence regularizes the latent space to follow a standard normal distribution.

    Args:
        reconstruction_loss: Type of reconstruction loss ('mse' or 'bce')
        beta: Weight for KL divergence term (beta-VAE)
        reduction: Reduction method ('mean', 'sum', or 'none')
    """

    def __init__(
        self,
        reconstruction_loss: str = 'mse',
        beta: float = 1.0,
        reduction: str = 'mean'
    ):
        super().__init__()
        if reconstruction_loss not in ['mse', 'bce']:
            raise ValueError(f"reconstruction_loss must be 'mse' or 'bce', got {reconstruction_loss}")
        if reduction not in ['mean', 'sum', 'none']:
            raise ValueError(f"reduction must be 'mean', 'sum', or 'none', got {reduction}")

        self.reconstruction_loss = reconstruction_loss
        self.beta = beta
        self.reduction = reduction

    def forward(self, recon_x: Tensor, x: Tensor, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Compute VAE loss.

        Args:
            recon_x: Reconstructed input of shape (batch_size, *)
            x: Original input of shape (batch_size, *)
            mu: Mean of latent distribution of shape (batch_size, latent_dim)
            logvar: Log-variance of latent distribution of shape (batch_size, latent_dim)

        Returns:
            Scalar loss tensor
        """
        xp = get_array_module(recon_x.data)

        # Reconstruction loss
        if self.reconstruction_loss == 'mse':
            # Mean squared error
            recon_loss_data = (recon_x.data - x.data) ** 2
            if self.reduction == 'mean':
                recon_loss_data = xp.mean(recon_loss_data)
            elif self.reduction == 'sum':
                recon_loss_data = xp.sum(recon_loss_data)
        else:  # bce
            # Binary cross entropy
            eps = 1e-7  # For numerical stability
            recon_x_clipped = xp.clip(recon_x.data, eps, 1 - eps)
            recon_loss_data = -(x.data * xp.log(recon_x_clipped) + (1 - x.data) * xp.log(1 - recon_x_clipped))
            if self.reduction == 'mean':
                recon_loss_data = xp.mean(recon_loss_data)
            elif self.reduction == 'sum':
                recon_loss_data = xp.sum(recon_loss_data)

        # KL divergence: -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
        # This measures how much the learned distribution deviates from standard normal
        kl_loss_data = -0.5 * xp.sum(1 + logvar.data - mu.data ** 2 - xp.exp(logvar.data), axis=1)
        if self.reduction == 'mean':
            kl_loss_data = xp.mean(kl_loss_data)
        elif self.reduction == 'sum':
            kl_loss_data = xp.sum(kl_loss_data)

        # Total loss
        total_loss_data = recon_loss_data + self.beta * kl_loss_data

        loss = Tensor(
            total_loss_data,
            requires_grad=recon_x.requires_grad or mu.requires_grad or logvar.requires_grad,
            device=recon_x.device,
            _children=(recon_x, x, mu, logvar),
            _op='vae_loss'
        )

        def _backward():
            if recon_x.requires_grad:
                # Gradient w.r.t. reconstructed output
                if self.reconstruction_loss == 'mse':
                    grad_recon = 2 * (recon_x.data - x.data)
                else:  # bce
                    eps = 1e-7
                    recon_x_clipped = xp.clip(recon_x.data, eps, 1 - eps)
                    grad_recon = -(x.data / recon_x_clipped - (1 - x.data) / (1 - recon_x_clipped))

                if self.reduction == 'mean':
                    grad_recon = grad_recon / recon_x.data.size
                elif self.reduction == 'none':
                    grad_recon = grad_recon * loss.grad

                recon_x.grad = grad_recon if recon_x.grad is None else recon_x.grad + grad_recon

            if mu.requires_grad:
                # Gradient w.r.t. mu: beta * mu
                grad_mu = self.beta * mu.data
                if self.reduction == 'mean':
                    grad_mu = grad_mu / mu.shape[0]
                elif self.reduction == 'none':
                    grad_mu = grad_mu * xp.expand_dims(loss.grad, axis=1)

                mu.grad = grad_mu if mu.grad is None else mu.grad + grad_mu

            if logvar.requires_grad:
                # Gradient w.r.t. logvar: -0.5 * beta * (1 - exp(logvar))
                grad_logvar = -0.5 * self.beta * (1 - xp.exp(logvar.data))
                if self.reduction == 'mean':
                    grad_logvar = grad_logvar / logvar.shape[0]
                elif self.reduction == 'none':
                    grad_logvar = grad_logvar * xp.expand_dims(loss.grad, axis=1)

                logvar.grad = grad_logvar if logvar.grad is None else logvar.grad + grad_logvar

        loss._backward = _backward
        return loss

    def __repr__(self):
        return (
            f"VAELoss(reconstruction_loss='{self.reconstruction_loss}', "
            f"beta={self.beta}, reduction='{self.reduction}')"
        )


class KLDivergenceLoss(Module):
    """
    KL Divergence Loss for VAE latent distributions.

    Computes the KL divergence between the learned latent distribution
    N(mu, sigma^2) and the standard normal distribution N(0, 1).

    KL(N(mu, sigma^2) || N(0, 1)) = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
                                   = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))

    Args:
        reduction: Reduction method ('mean', 'sum', or 'none')
    """

    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        if reduction not in ['mean', 'sum', 'none']:
            raise ValueError(f"reduction must be 'mean', 'sum', or 'none', got {reduction}")
        self.reduction = reduction

    def forward(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Compute KL divergence.

        Args:
            mu: Mean of latent distribution of shape (batch_size, latent_dim)
            logvar: Log-variance of latent distribution of shape (batch_size, latent_dim)

        Returns:
            KL divergence loss
        """
        xp = get_array_module(mu.data)

        # KL divergence: -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
        kl_data = -0.5 * xp.sum(1 + logvar.data - mu.data ** 2 - xp.exp(logvar.data), axis=1)

        if self.reduction == 'mean':
            kl_data = xp.mean(kl_data)
        elif self.reduction == 'sum':
            kl_data = xp.sum(kl_data)

        loss = Tensor(
            kl_data,
            requires_grad=mu.requires_grad or logvar.requires_grad,
            device=mu.device,
            _children=(mu, logvar),
            _op='kl_divergence'
        )

        def _backward():
            batch_size = mu.shape[0]

            if mu.requires_grad:
                # Gradient w.r.t. mu: mu
                grad_mu = mu.data
                if self.reduction == 'mean':
                    grad_mu = grad_mu / batch_size
                elif self.reduction == 'none':
                    grad_mu = grad_mu * xp.expand_dims(loss.grad, axis=1)

                mu.grad = grad_mu if mu.grad is None else mu.grad + grad_mu

            if logvar.requires_grad:
                # Gradient w.r.t. logvar: -0.5 * (1 - exp(logvar))
                grad_logvar = -0.5 * (1 - xp.exp(logvar.data))
                if self.reduction == 'mean':
                    grad_logvar = grad_logvar / batch_size
                elif self.reduction == 'none':
                    grad_logvar = grad_logvar * xp.expand_dims(loss.grad, axis=1)

                logvar.grad = grad_logvar if logvar.grad is None else logvar.grad + grad_logvar

        loss._backward = _backward
        return loss

    def __repr__(self):
        return f"KLDivergenceLoss(reduction='{self.reduction}')"
