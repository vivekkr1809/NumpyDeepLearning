# Transformer Architecture & Tiny Shakespeare Example

This directory contains a complete implementation of the Transformer architecture with a practical example using Tiny Shakespeare dataset.

## Overview

The Transformer, introduced in ["Attention is All You Need"](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017), revolutionized sequence modeling by replacing recurrence with attention mechanisms.

### Key Features

- **Self-Attention**: Allows each position to attend to all positions
- **Multi-Head Attention**: Parallel attention computations for diverse patterns
- **Positional Encoding**: Injects position information without recurrence
- **Feed-Forward Networks**: Position-wise transformations
- **Layer Normalization**: Stabilizes training
- **Residual Connections**: Enables deep architectures

## Architecture Components

### 1. Attention Mechanisms (`numpy_dl/nn/attention.py`)

```python
from numpy_dl.nn import MultiHeadAttention, PositionalEncoding

# Multi-head attention
attention = MultiHeadAttention(d_model=512, num_heads=8, dropout=0.1)
output, attn_weights = attention(query, key, value, mask=None)

# Positional encoding
pos_enc = PositionalEncoding(d_model=512, max_len=5000)
x_with_pos = pos_enc(x)
```

### 2. Transformer Models (`numpy_dl/models/transformer.py`)

#### GPT-Style (Decoder-Only) for Language Modeling

```python
from numpy_dl.models import GPTModel

model = GPTModel(
    vocab_size=10000,
    d_model=256,
    num_heads=4,
    num_layers=4,
    d_ff=1024,
    max_seq_len=512,
    dropout=0.1
)

# Forward pass
logits = model(input_tokens)  # (batch, seq_len, vocab_size)

# Generate text
generated = model.generate(
    start_tokens=seed_tokens,
    max_new_tokens=100,
    temperature=0.8
)
```

#### Full Transformer (Encoder-Decoder) for Seq2Seq

```python
from numpy_dl.models import Transformer

model = Transformer(
    src_vocab_size=10000,
    tgt_vocab_size=10000,
    d_model=512,
    num_heads=8,
    num_encoder_layers=6,
    num_decoder_layers=6,
    d_ff=2048,
    max_seq_len=5000,
    dropout=0.1
)

# Forward pass
output = model(src, tgt, src_mask=None, tgt_mask=causal_mask)
```

## Tiny Shakespeare Example

### Quick Start

```bash
# Run the example
python examples/nlp/shakespeare_transformer.py
```

### What It Does

1. **Downloads Data**: Automatically downloads Tiny Shakespeare (~1MB)
2. **Tokenizes**: Creates character-level vocabulary (~65 chars)
3. **Trains Model**: GPT-style transformer learns to predict next character
4. **Generates Text**: Produces Shakespeare-like text samples

### Model Configuration

```python
config = {
    'seq_len': 64,          # Sequence length
    'd_model': 128,         # Model dimension
    'num_heads': 4,         # Attention heads
    'num_layers': 3,        # Transformer layers
    'batch_size': 32,       # Batch size
    'epochs': 10,           # Training epochs
    'learning_rate': 0.001  # Learning rate
}
```

### Expected Output

After training, the model generates text like:

```
First Citizen:
Before we proceed any further, hear me speak.

All:
Speak, speak.

First Citizen:
You are all resolved rather to die than to famish?

All:
Resolved. resolved.
```

## Training Process

### 1. Data Preparation

```python
from numpy_dl.core.tensor import Tensor
from numpy_dl.data import TensorDataset, DataLoader

# Create sequences
inputs, targets, tokenizer = create_dataset(text, seq_len=64)

# Create data loader
dataset = TensorDataset(Tensor(inputs), Tensor(targets))
loader = DataLoader(dataset, batch_size=32, shuffle=True)
```

### 2. Model Training

```python
from numpy_dl.models import GPTModel
from numpy_dl.optim import Adam
from numpy_dl.loss import CrossEntropyLoss

# Initialize
model = GPTModel(vocab_size=vocab_size, d_model=128, num_heads=4, num_layers=3)
optimizer = Adam(model.parameters(), lr=0.001)
criterion = CrossEntropyLoss()

# Training loop
model.train()
for epoch in range(epochs):
    for inputs, targets in loader:
        # Forward
        logits = model(inputs)
        loss = criterion(logits.reshape(-1, vocab_size), targets.reshape(-1))

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### 3. Text Generation

```python
# Generate text
model.eval()
seed = "ROMEO:\n"
seed_tokens = Tensor(np.array([tokenizer.encode(seed)]))

generated = model.generate(
    start_tokens=seed_tokens,
    max_new_tokens=200,
    temperature=0.8  # Higher = more creative
)

text = tokenizer.decode(generated.numpy()[0].tolist())
print(text)
```

## Hyperparameter Tuning

### Model Size

| Parameter | Small | Medium | Large |
|-----------|-------|--------|-------|
| d_model | 128 | 256 | 512 |
| num_heads | 4 | 8 | 16 |
| num_layers | 2 | 4 | 6 |
| d_ff | 512 | 1024 | 2048 |

### Training

- **Learning Rate**: Start with 0.001, use scheduler for longer training
- **Batch Size**: 32-128 depending on sequence length and memory
- **Sequence Length**: 64-512 characters/tokens
- **Epochs**: 10-50 for small datasets

### Generation

- **Temperature**:
  - 0.5 = Conservative (repetitive)
  - 1.0 = Balanced
  - 1.5 = Creative (diverse but may be incoherent)

## Advanced Usage

### Custom Tokenization

```python
class CustomTokenizer:
    def __init__(self, vocab):
        self.vocab = vocab
        self.token_to_idx = {t: i for i, t in enumerate(vocab)}
        self.idx_to_token = {i: t for i, t in enumerate(vocab)}

    def encode(self, text):
        # Your tokenization logic
        return [self.token_to_idx[t] for t in tokens]

    def decode(self, indices):
        return ''.join([self.idx_to_token[i] for i in indices])
```

### Attention Visualization

```python
# During forward pass
output, attn_weights = model.layers[0].self_attn(x, x, x, mask)

# attn_weights shape: (batch, num_heads, seq_len, seq_len)
# Visualize attention patterns
import matplotlib.pyplot as plt
plt.imshow(attn_weights[0, 0].numpy())
plt.colorbar()
plt.show()
```

### Transfer Learning

```python
# Train on large corpus
model = GPTModel(vocab_size=large_vocab, ...)
# ... training ...

# Fine-tune on specific domain
for param in model.parameters():
    param.requires_grad = True  # All params trainable

# Or freeze early layers
for layer in model.layers[:2]:
    for param in layer.parameters():
        param.requires_grad = False
```

## Comparison with Other Architectures

### vs RNN/LSTM

| Feature | Transformer | RNN/LSTM |
|---------|-------------|----------|
| Parallelization | ✅ Full | ❌ Sequential |
| Long-range deps | ✅ O(1) | ⚠️ O(n) |
| Training speed | ✅ Fast | ⚠️ Slow |
| Memory | ⚠️ O(n²) | ✅ O(n) |

### vs CNN

| Feature | Transformer | CNN |
|---------|-------------|-----|
| Sequence modeling | ✅ Native | ⚠️ Limited |
| Position encoding | ✅ Explicit | ✅ Implicit |
| Long-range deps | ✅ Direct | ⚠️ Receptive field |
| Computational cost | ⚠️ O(n²) | ✅ O(n) |

## Applications

### Implemented
- **Text Generation**: Character-level language modeling (Shakespeare)
- **Next Token Prediction**: Autoregressive modeling

### Possible Extensions
- **Machine Translation**: Seq2Seq with encoder-decoder
- **Text Classification**: Add classification head
- **Named Entity Recognition**: Token-level classification
- **Question Answering**: Context-question-answer triplets
- **Summarization**: Long text to short summary
- **Code Generation**: Train on code corpus

## Performance Tips

### Memory Optimization
- Reduce sequence length
- Use gradient checkpointing (save/recompute activations)
- Smaller batch sizes for longer sequences
- Use mixed precision (if supported)

### Training Speedup
- Gradient accumulation for effective larger batches
- Learning rate warmup for stable training
- Label smoothing for better generalization
- Dropout for regularization

### Generation Quality
- Nucleus/top-p sampling instead of greedy
- Beam search for better quality
- Length penalties to control generation
- Repetition penalties to avoid loops

## Troubleshooting

### Common Issues

**NaN Loss**
- Check learning rate (try lower)
- Use gradient clipping
- Check for numerical instability in attention

**Poor Generation Quality**
- Train longer (more epochs)
- Increase model size
- Lower temperature for more conservative output
- Check if model is overfitting

**Slow Training**
- Reduce sequence length
- Smaller batch size
- Fewer layers/smaller d_model
- Enable CPU-specific optimizations

## References

- Vaswani et al. (2017). "Attention is All You Need"
- Radford et al. (2018). "Improving Language Understanding by Generative Pre-Training" (GPT)
- Devlin et al. (2018). "BERT: Pre-training of Deep Bidirectional Transformers"
- Brown et al. (2020). "Language Models are Few-Shot Learners" (GPT-3)

## Citation

If you use this implementation, please cite:

```bibtex
@software{numpy_dl_transformer,
  title = {NumPy Deep Learning Transformer Implementation},
  author = {NumPy DL Contributors},
  year = {2025},
  url = {https://github.com/vivekkr1809/NumpyDeepLearning}
}
```
