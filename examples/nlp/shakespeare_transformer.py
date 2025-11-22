"""
Tiny Shakespeare Text Generation with GPT-style Transformer.

This example demonstrates:
1. Character-level language modeling
2. GPT-style (decoder-only) transformer architecture
3. Autoregressive text generation
4. Training on Shakespeare's works

The model learns to generate Shakespeare-like text by predicting the next character
given previous characters.

Dataset: Tiny Shakespeare (https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt)
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import logging
from pathlib import Path

# Import framework components
from numpy_dl.core.tensor import Tensor
from numpy_dl.models.transformer import GPTModel
from numpy_dl.optim import Adam
from numpy_dl.loss import CrossEntropyLoss
from numpy_dl.utils.logging import configure_logging, get_logger
from numpy_dl.data import DataLoader, TensorDataset


def download_shakespeare():
    """Download Tiny Shakespeare dataset if not present."""
    data_path = Path('data/shakespeare.txt')

    if data_path.exists():
        print(f"Dataset already exists at {data_path}")
        return str(data_path)

    print("Downloading Tiny Shakespeare dataset...")
    data_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import urllib.request
        url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
        urllib.request.urlretrieve(url, data_path)
        print(f"Downloaded to {data_path}")
    except Exception as e:
        print(f"Error downloading: {e}")
        print("Creating sample data instead...")

        # Create sample Shakespeare-like text
        sample_text = """First Citizen:
Before we proceed any further, hear me speak.

All:
Speak, speak.

First Citizen:
You are all resolved rather to die than to famish?

All:
Resolved. resolved.

First Citizen:
First, you know Caius Marcius is chief enemy to the people.

All:
We know't, we know't.

First Citizen:
Let us kill him, and we'll have corn at our own price.
Is't a verdict?

All:
No more talking on't; let it be done: away, away!

Second Citizen:
One word, good citizens.

First Citizen:
We are accounted poor citizens, the patricians good.
What authority surfeits on would relieve us: if they
would yield us but the superfluity, while it were
wholesome, we might guess they relieved us humanely;
but they think we are too dear: the leanness that
afflicts us, the object of our misery, is as an
inventory to particularise their abundance; our
sufferance is a gain to them Let us revenge this with
our pikes, ere we become rakes: for the gods know I
speak this in hunger for bread, not in thirst for revenge.
""" * 20  # Repeat to have more training data

        with open(data_path, 'w') as f:
            f.write(sample_text)

    return str(data_path)


class CharacterTokenizer:
    """Simple character-level tokenizer."""

    def __init__(self, text: str):
        """
        Initialize tokenizer from text.

        Args:
            text: Training text
        """
        # Get unique characters
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)

        # Create mappings
        self.char_to_idx = {ch: i for i, ch in enumerate(chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(chars)}

        print(f"Vocabulary size: {self.vocab_size}")
        print(f"Characters: {''.join(chars)}")

    def encode(self, text: str) -> list:
        """Convert text to list of indices."""
        return [self.char_to_idx[ch] for ch in text]

    def decode(self, indices: list) -> str:
        """Convert list of indices to text."""
        return ''.join([self.idx_to_char[i] for i in indices])


def create_dataset(text: str, seq_len: int = 64):
    """
    Create training dataset from text.

    Args:
        text: Input text
        seq_len: Sequence length for each training example

    Returns:
        (input_indices, target_indices) as numpy arrays
    """
    # Convert text to indices
    tokenizer = CharacterTokenizer(text)
    data = tokenizer.encode(text)

    # Create sequences
    inputs = []
    targets = []

    for i in range(0, len(data) - seq_len):
        inputs.append(data[i:i + seq_len])
        targets.append(data[i + 1:i + seq_len + 1])

    return np.array(inputs), np.array(targets), tokenizer


def train_shakespeare_model(
    data_path: str,
    seq_len: int = 64,
    d_model: int = 128,
    num_heads: int = 4,
    num_layers: int = 3,
    batch_size: int = 32,
    epochs: int = 10,
    learning_rate: float = 0.001,
    device: str = 'cpu'
):
    """
    Train a GPT-style transformer on Shakespeare text.

    Args:
        data_path: Path to text file
        seq_len: Sequence length
        d_model: Model dimension
        num_heads: Number of attention heads
        num_layers: Number of transformer layers
        batch_size: Batch size
        epochs: Number of training epochs
        learning_rate: Learning rate
        device: Device to train on
    """
    logger = get_logger('training')

    # Load data
    logger.info("Loading dataset", path=data_path)
    with open(data_path, 'r') as f:
        text = f.read()

    logger.info("Dataset loaded", total_chars=len(text))

    # Create dataset
    logger.info("Creating training dataset", seq_len=seq_len)
    inputs, targets, tokenizer = create_dataset(text, seq_len)

    logger.info(
        "Dataset created",
        num_sequences=len(inputs),
        vocab_size=tokenizer.vocab_size
    )

    # Create data loaders
    input_tensor = Tensor(inputs)
    target_tensor = Tensor(targets)
    dataset = TensorDataset(input_tensor, target_tensor)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Create model
    logger.info("Creating GPT model")
    model = GPTModel(
        vocab_size=tokenizer.vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_model * 4,
        max_seq_len=seq_len + 100,  # Allow for generation
        dropout=0.1
    )

    # Loss and optimizer
    criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    logger.info(
        "Starting training",
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate
    )

    # Training loop
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, (batch_inputs, batch_targets) in enumerate(train_loader):
            # Forward pass
            logits = model(batch_inputs)  # (batch, seq_len, vocab_size)

            # Reshape for loss calculation
            batch_size_actual = logits.shape[0]
            logits_flat = logits.reshape(-1, tokenizer.vocab_size)
            targets_flat = batch_targets.reshape(-1)

            # Compute loss
            loss = criterion(logits_flat, targets_flat)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track metrics
            epoch_loss += float(loss.data)
            num_batches += 1

            # Log progress
            if (batch_idx + 1) % 10 == 0:
                avg_loss = epoch_loss / num_batches
                print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx+1}/{len(train_loader)}, "
                      f"Loss: {float(loss.data):.4f}, Avg Loss: {avg_loss:.4f}")

        avg_loss = epoch_loss / num_batches
        logger.info(f"Epoch {epoch+1} completed", avg_loss=avg_loss)
        print(f"\nEpoch {epoch+1}/{epochs} - Average Loss: {avg_loss:.4f}\n")

        # Generate sample text
        if (epoch + 1) % 2 == 0:  # Generate every 2 epochs
            print("\n" + "="*80)
            print("SAMPLE GENERATION:")
            print("="*80)
            generate_text(model, tokenizer, seed_text="First Citizen:\n", max_new_tokens=200)
            print("="*80 + "\n")

    logger.info("Training completed")

    return model, tokenizer


def generate_text(
    model: GPTModel,
    tokenizer: CharacterTokenizer,
    seed_text: str = "ROMEO:\n",
    max_new_tokens: int = 300,
    temperature: float = 0.8
):
    """
    Generate text using the trained model.

    Args:
        model: Trained GPT model
        tokenizer: Character tokenizer
        seed_text: Seed text to start generation
        max_new_tokens: Number of new tokens to generate
        temperature: Sampling temperature
    """
    model.eval()

    # Encode seed text
    seed_indices = tokenizer.encode(seed_text)
    start_tokens = Tensor(np.array([seed_indices]))

    # Generate
    print(f"Seed: {seed_text}")
    print("-" * 40)

    generated = model.generate(
        start_tokens,
        max_new_tokens=max_new_tokens,
        temperature=temperature
    )

    # Decode and print
    generated_text = tokenizer.decode(generated.numpy()[0].tolist())
    print(generated_text)

    return generated_text


def main():
    """Main training script."""
    print("\n" + "="*80)
    print("TINY SHAKESPEARE TEXT GENERATION WITH TRANSFORMER")
    print("="*80 + "\n")

    # Configure logging
    configure_logging(
        level=logging.INFO,
        console=True,
        log_dir='./logs'
    )

    # Download dataset
    data_path = download_shakespeare()

    # Training hyperparameters
    config = {
        'seq_len': 64,        # Sequence length
        'd_model': 128,       # Model dimension
        'num_heads': 4,       # Attention heads
        'num_layers': 3,      # Transformer layers
        'batch_size': 32,     # Batch size
        'epochs': 10,         # Training epochs
        'learning_rate': 0.001,  # Learning rate
        'device': 'cpu'
    }

    print("Training Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()

    # Train model
    model, tokenizer = train_shakespeare_model(data_path, **config)

    # Interactive generation
    print("\n" + "="*80)
    print("FINAL TEXT GENERATION")
    print("="*80 + "\n")

    # Generate samples with different seeds
    seeds = [
        "First Citizen:\n",
        "ROMEO:\n",
        "The ",
        "What "
    ]

    for seed in seeds:
        print("\n" + "-"*80)
        generate_text(model, tokenizer, seed_text=seed, max_new_tokens=200, temperature=0.8)
        print("-"*80)

    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80 + "\n")

    print("The model has learned to generate Shakespeare-like text.")
    print("Try experimenting with:")
    print("  - Different temperatures (0.5 = conservative, 1.5 = creative)")
    print("  - Different seed texts")
    print("  - Longer training (more epochs)")
    print("  - Larger model (more layers/dimensions)")


if __name__ == "__main__":
    main()
