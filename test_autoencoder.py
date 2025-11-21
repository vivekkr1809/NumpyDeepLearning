"""
Comprehensive tests for Autoencoder models.

Tests all autoencoder variants including standard autoencoders,
convolutional autoencoders, variational autoencoders, and their loss functions.
"""

import sys
import numpy as np

print("=" * 80)
print("Testing Autoencoder Models and Components")
print("=" * 80)

# Test 1: Import all autoencoder components
print("\n[1/12] Testing imports...")
try:
    from numpy_dl.models import (
        Autoencoder,
        ConvAutoencoder,
        VariationalAutoencoder,
        ConvVariationalAutoencoder
    )
    from numpy_dl.loss import VAELoss, KLDivergenceLoss
    from numpy_dl.nn import ConvTranspose2d
    import numpy_dl as ndl
    print("✓ All autoencoder components imported successfully")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Test Autoencoder instantiation
print("\n[2/12] Testing Autoencoder instantiation...")
try:
    ae = Autoencoder(input_size=784, hidden_sizes=[512, 256], latent_dim=64)
    print(f"✓ Autoencoder created: {ae}")
    assert ae.input_size == 784
    assert ae.latent_dim == 64
    print("✓ Autoencoder parameters verified")
except Exception as e:
    print(f"✗ Autoencoder instantiation failed: {e}")
    sys.exit(1)

# Test 3: Test Autoencoder forward pass
print("\n[3/12] Testing Autoencoder forward pass...")
try:
    batch_size = 8
    x = ndl.tensor(np.random.randn(batch_size, 784).astype(np.float32), requires_grad=True)

    # Test encoding
    z = ae.encode(x)
    assert z.shape == (batch_size, 64), f"Expected shape (8, 64), got {z.shape}"
    print(f"✓ Encoding output shape: {z.shape}")

    # Test decoding
    recon = ae.decode(z)
    assert recon.shape == (batch_size, 784), f"Expected shape (8, 784), got {recon.shape}"
    print(f"✓ Decoding output shape: {recon.shape}")

    # Test full forward pass
    output = ae(x)
    assert output.shape == x.shape, f"Expected shape {x.shape}, got {output.shape}"
    print(f"✓ Full forward pass output shape: {output.shape}")
except Exception as e:
    print(f"✗ Autoencoder forward pass failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Test Autoencoder backward pass
print("\n[4/12] Testing Autoencoder backward pass...")
try:
    from numpy_dl.loss import MSELoss
    criterion = MSELoss()

    x = ndl.tensor(np.random.rand(4, 784).astype(np.float32), requires_grad=False)
    output = ae(x)
    loss = criterion(output, x)

    loss.backward()

    # Check that gradients exist
    params_with_grad = sum(1 for p in ae.parameters() if p.grad is not None)
    print(f"✓ Backward pass successful, {params_with_grad} parameters have gradients")
    assert params_with_grad > 0, "No parameters received gradients"
except Exception as e:
    print(f"✗ Autoencoder backward pass failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Test ConvAutoencoder
print("\n[5/12] Testing ConvAutoencoder...")
try:
    conv_ae = ConvAutoencoder(in_channels=1, base_channels=32, latent_dim=128, image_size=28)
    print(f"✓ ConvAutoencoder created: {conv_ae}")

    # Test forward pass with image data
    batch_size = 4
    x_img = ndl.tensor(np.random.rand(batch_size, 1, 28, 28).astype(np.float32), requires_grad=False)

    z = conv_ae.encode(x_img)
    assert z.shape == (batch_size, 128), f"Expected shape (4, 128), got {z.shape}"
    print(f"✓ ConvAutoencoder encoding shape: {z.shape}")

    recon = conv_ae.decode(z)
    print(f"✓ ConvAutoencoder decoding shape: {recon.shape}")

    output = conv_ae(x_img)
    print(f"✓ ConvAutoencoder forward pass shape: {output.shape}")
except Exception as e:
    print(f"✗ ConvAutoencoder failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Test VariationalAutoencoder
print("\n[6/12] Testing VariationalAutoencoder...")
try:
    vae = VariationalAutoencoder(input_size=784, hidden_sizes=[512, 256], latent_dim=64)
    print(f"✓ VAE created: {vae}")

    batch_size = 8
    x = ndl.tensor(np.random.rand(batch_size, 784).astype(np.float32), requires_grad=False)

    # Test encoding (returns mu and logvar)
    mu, logvar = vae.encode(x)
    assert mu.shape == (batch_size, 64), f"Expected mu shape (8, 64), got {mu.shape}"
    assert logvar.shape == (batch_size, 64), f"Expected logvar shape (8, 64), got {logvar.shape}"
    print(f"✓ VAE encoding outputs: mu{mu.shape}, logvar{logvar.shape}")

    # Test reparameterization
    z = vae.reparameterize(mu, logvar)
    assert z.shape == (batch_size, 64), f"Expected z shape (8, 64), got {z.shape}"
    print(f"✓ VAE reparameterization output: {z.shape}")

    # Test full forward pass (returns recon, mu, logvar)
    recon, mu, logvar = vae(x)
    assert recon.shape == x.shape, f"Expected recon shape {x.shape}, got {recon.shape}"
    print(f"✓ VAE forward pass outputs: recon{recon.shape}, mu{mu.shape}, logvar{logvar.shape}")
except Exception as e:
    print(f"✗ VariationalAutoencoder failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 7: Test VAE sampling
print("\n[7/12] Testing VAE sampling...")
try:
    samples = vae.sample(num_samples=5, device='cpu')
    assert samples.shape == (5, 784), f"Expected shape (5, 784), got {samples.shape}"
    print(f"✓ VAE sampling successful: {samples.shape}")
except Exception as e:
    print(f"✗ VAE sampling failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 8: Test ConvVariationalAutoencoder
print("\n[8/12] Testing ConvVariationalAutoencoder...")
try:
    conv_vae = ConvVariationalAutoencoder(in_channels=1, base_channels=32, latent_dim=128, image_size=28)
    print(f"✓ ConvVAE created: {conv_vae}")

    batch_size = 4
    x_img = ndl.tensor(np.random.rand(batch_size, 1, 28, 28).astype(np.float32), requires_grad=False)

    mu, logvar = conv_vae.encode(x_img)
    assert mu.shape == (batch_size, 128), f"Expected mu shape (4, 128), got {mu.shape}"
    print(f"✓ ConvVAE encoding outputs: mu{mu.shape}, logvar{logvar.shape}")

    recon, mu, logvar = conv_vae(x_img)
    print(f"✓ ConvVAE forward pass outputs: recon{recon.shape}, mu{mu.shape}, logvar{logvar.shape}")

    samples = conv_vae.sample(num_samples=3, device='cpu')
    print(f"✓ ConvVAE sampling successful: {samples.shape}")
except Exception as e:
    print(f"✗ ConvVariationalAutoencoder failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 9: Test VAELoss with MSE reconstruction
print("\n[9/12] Testing VAELoss with MSE reconstruction...")
try:
    vae_loss_mse = VAELoss(reconstruction_loss='mse', beta=1.0, reduction='mean')
    print(f"✓ VAELoss (MSE) created: {vae_loss_mse}")

    x = ndl.tensor(np.random.rand(4, 784).astype(np.float32), requires_grad=False)
    recon, mu, logvar = vae(x)

    loss = vae_loss_mse(recon, x, mu, logvar)
    print(f"✓ VAELoss (MSE) computed: {loss.data}")
    assert loss.shape == (), f"Expected scalar loss, got shape {loss.shape}"
except Exception as e:
    print(f"✗ VAELoss (MSE) failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 10: Test VAELoss with BCE reconstruction
print("\n[10/12] Testing VAELoss with BCE reconstruction...")
try:
    vae_loss_bce = VAELoss(reconstruction_loss='bce', beta=0.5, reduction='mean')
    print(f"✓ VAELoss (BCE) created: {vae_loss_bce}")

    x = ndl.tensor(np.random.rand(4, 784).astype(np.float32), requires_grad=False)
    recon, mu, logvar = vae(x)

    loss = vae_loss_bce(recon, x, mu, logvar)
    print(f"✓ VAELoss (BCE) computed: {loss.data}")
    assert loss.shape == (), f"Expected scalar loss, got shape {loss.shape}"
except Exception as e:
    print(f"✗ VAELoss (BCE) failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 11: Test KLDivergenceLoss
print("\n[11/12] Testing KLDivergenceLoss...")
try:
    kl_loss = KLDivergenceLoss(reduction='mean')
    print(f"✓ KLDivergenceLoss created: {kl_loss}")

    x = ndl.tensor(np.random.rand(4, 784).astype(np.float32), requires_grad=False)
    mu, logvar = vae.encode(x)

    kl = kl_loss(mu, logvar)
    print(f"✓ KL divergence computed: {kl.data}")
    assert kl.shape == (), f"Expected scalar loss, got shape {kl.shape}"
    assert kl.data >= 0, "KL divergence should be non-negative"
    print("✓ KL divergence is non-negative")
except Exception as e:
    print(f"✗ KLDivergenceLoss failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 12: Test ConvTranspose2d
print("\n[12/12] Testing ConvTranspose2d layer...")
try:
    conv_transpose = ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1)
    print(f"✓ ConvTranspose2d created: {conv_transpose}")

    x = ndl.tensor(np.random.randn(2, 64, 7, 7).astype(np.float32), requires_grad=False)
    output = conv_transpose(x)
    print(f"✓ ConvTranspose2d forward pass: input {x.shape} -> output {output.shape}")

    # Expected output size: (H-1)*stride - 2*padding + kernel + output_padding
    # (7-1)*2 - 2*1 + 4 + 0 = 12 - 2 + 4 = 14
    expected_h = expected_w = 14
    assert output.shape[2] == expected_h and output.shape[3] == expected_w, \
        f"Expected spatial size {expected_h}x{expected_w}, got {output.shape[2]}x{output.shape[3]}"
    print(f"✓ ConvTranspose2d output dimensions correct: {output.shape}")
except Exception as e:
    print(f"✗ ConvTranspose2d failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 13: End-to-end VAE training test
print("\n[BONUS] End-to-end VAE training test...")
try:
    from numpy_dl.optim import Adam

    # Create small VAE
    vae_small = VariationalAutoencoder(input_size=100, hidden_sizes=[64], latent_dim=16)
    optimizer = Adam(vae_small.parameters(), lr=0.001)
    criterion = VAELoss(reconstruction_loss='mse', beta=1.0)

    # Generate synthetic data
    x = ndl.tensor(np.random.rand(8, 100).astype(np.float32), requires_grad=False)

    # Training step
    vae_small.train()
    recon, mu, logvar = vae_small(x)
    loss = criterion(recon, x, mu, logvar)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"✓ End-to-end VAE training successful, loss: {loss.data}")
except Exception as e:
    print(f"✗ End-to-end VAE training failed: {e}")
    import traceback
    traceback.print_exc()
    # Don't exit, this is a bonus test

print("\n" + "=" * 80)
print("All autoencoder tests passed successfully!")
print("=" * 80)
