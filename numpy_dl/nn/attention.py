"""Attention mechanisms for Transformer models."""

import numpy as np
from numpy_dl.core.module import Module
from numpy_dl.core.tensor import Tensor
from numpy_dl.nn import Linear, Dropout
from numpy_dl.utils.logging import get_logger
from typing import Optional


class ScaledDotProductAttention(Module):
    """
    Scaled Dot-Product Attention.

    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V

    Args:
        dropout: Dropout probability (default: 0.1)

    Example:
        >>> attention = ScaledDotProductAttention(dropout=0.1)
        >>> output, attn_weights = attention(query, key, value, mask=None)
    """

    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.dropout = Dropout(dropout) if dropout > 0 else None
        self.logger = get_logger('transformer')

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Optional[Tensor] = None
    ) -> tuple:
        """
        Forward pass for scaled dot-product attention.

        Args:
            query: Query tensor (batch, ..., seq_len, d_k)
            key: Key tensor (batch, ..., seq_len, d_k)
            value: Value tensor (batch, ..., seq_len, d_v)
            mask: Optional mask tensor (batch, ..., seq_len, seq_len)

        Returns:
            (output, attention_weights)
        """
        # Get dimensionality for scaling
        d_k = query.shape[-1]

        # Compute attention scores: QK^T / sqrt(d_k)
        scores = query @ key.transpose(-2, -1)  # (..., seq_len, seq_len)
        scores = scores / np.sqrt(d_k)

        # Apply mask if provided (for padding or causal masking)
        if mask is not None:
            # Use a very large negative value instead of -inf to avoid NaN
            scores = scores + (mask * -1e9)

        # Apply softmax to get attention weights
        # Manual softmax implementation
        exp_scores = scores.exp()
        attention_weights = exp_scores / exp_scores.sum(axis=-1, keepdims=True)

        # Apply dropout to attention weights
        if self.dropout is not None and self.training:
            attention_weights = self.dropout(attention_weights)

        # Apply attention to values
        output = attention_weights @ value

        return output, attention_weights


class MultiHeadAttention(Module):
    """
    Multi-Head Attention mechanism.

    MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O
    where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)

    Args:
        d_model: Dimension of the model
        num_heads: Number of attention heads
        dropout: Dropout probability (default: 0.1)

    Example:
        >>> mha = MultiHeadAttention(d_model=512, num_heads=8)
        >>> output, attn_weights = mha(query, key, value, mask=None)
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.logger = get_logger('transformer')

        # Linear projections for Q, K, V
        self.W_q = Linear(d_model, d_model)
        self.W_k = Linear(d_model, d_model)
        self.W_v = Linear(d_model, d_model)

        # Output projection
        self.W_o = Linear(d_model, d_model)

        # Attention mechanism
        self.attention = ScaledDotProductAttention(dropout)

        self.dropout = Dropout(dropout) if dropout > 0 else None

    def _split_heads(self, x: Tensor) -> Tensor:
        """
        Split the last dimension into (num_heads, d_k).

        Args:
            x: Tensor of shape (batch, seq_len, d_model)

        Returns:
            Tensor of shape (batch, num_heads, seq_len, d_k)
        """
        batch_size = x.shape[0]
        seq_len = x.shape[1]

        # Reshape to (batch, seq_len, num_heads, d_k)
        x = x.reshape(batch_size, seq_len, self.num_heads, self.d_k)

        # Transpose to (batch, num_heads, seq_len, d_k)
        x = x.transpose(0, 2, 1, 3)

        return x

    def _combine_heads(self, x: Tensor) -> Tensor:
        """
        Combine heads back to original dimension.

        Args:
            x: Tensor of shape (batch, num_heads, seq_len, d_k)

        Returns:
            Tensor of shape (batch, seq_len, d_model)
        """
        batch_size = x.shape[0]
        seq_len = x.shape[2]

        # Transpose to (batch, seq_len, num_heads, d_k)
        x = x.transpose(0, 2, 1, 3)

        # Reshape to (batch, seq_len, d_model)
        x = x.reshape(batch_size, seq_len, self.d_model)

        return x

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Optional[Tensor] = None
    ) -> tuple:
        """
        Forward pass for multi-head attention.

        Args:
            query: Query tensor (batch, seq_len, d_model)
            key: Key tensor (batch, seq_len, d_model)
            value: Value tensor (batch, seq_len, d_model)
            mask: Optional mask tensor

        Returns:
            (output, attention_weights)
        """
        batch_size = query.shape[0]

        # Linear projections
        Q = self.W_q(query)  # (batch, seq_len, d_model)
        K = self.W_k(key)    # (batch, seq_len, d_model)
        V = self.W_v(value)  # (batch, seq_len, d_model)

        # Split into multiple heads
        Q = self._split_heads(Q)  # (batch, num_heads, seq_len, d_k)
        K = self._split_heads(K)  # (batch, num_heads, seq_len, d_k)
        V = self._split_heads(V)  # (batch, num_heads, seq_len, d_k)

        # Apply attention
        attn_output, attn_weights = self.attention(Q, K, V, mask)

        # Combine heads
        output = self._combine_heads(attn_output)  # (batch, seq_len, d_model)

        # Final linear projection
        output = self.W_o(output)

        if self.dropout is not None:
            output = self.dropout(output)

        return output, attn_weights


class PositionwiseFeedForward(Module):
    """
    Position-wise Feed-Forward Network.

    FFN(x) = max(0, xW_1 + b_1)W_2 + b_2

    Args:
        d_model: Dimension of the model
        d_ff: Dimension of the feed-forward layer
        dropout: Dropout probability (default: 0.1)

    Example:
        >>> ffn = PositionwiseFeedForward(d_model=512, d_ff=2048)
        >>> output = ffn(x)
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()

        self.linear1 = Linear(d_model, d_ff)
        self.linear2 = Linear(d_ff, d_model)
        self.dropout = Dropout(dropout) if dropout > 0 else None

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (batch, seq_len, d_model)

        Returns:
            Output tensor (batch, seq_len, d_model)
        """
        # First linear + ReLU
        x = self.linear1(x)

        # Manual ReLU
        x_data = x.data
        x_data = np.maximum(0, x_data)
        x = Tensor(x_data, requires_grad=x.requires_grad, device=x.device)

        if self.dropout is not None:
            x = self.dropout(x)

        # Second linear
        x = self.linear2(x)

        if self.dropout is not None:
            x = self.dropout(x)

        return x


class PositionalEncoding(Module):
    """
    Positional Encoding using sine and cosine functions.

    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Args:
        d_model: Dimension of the model
        max_len: Maximum sequence length (default: 5000)
        dropout: Dropout probability (default: 0.1)

    Example:
        >>> pos_enc = PositionalEncoding(d_model=512, max_len=1000)
        >>> output = pos_enc(x)
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()

        self.dropout = Dropout(dropout) if dropout > 0 else None

        # Create positional encoding matrix
        pe = np.zeros((max_len, d_model))
        position = np.arange(0, max_len).reshape(-1, 1)
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))

        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)

        # Store as non-trainable parameter
        self.pe = pe  # (max_len, d_model)

    def forward(self, x: Tensor) -> Tensor:
        """
        Add positional encoding to input.

        Args:
            x: Input tensor (batch, seq_len, d_model)

        Returns:
            Output tensor with positional encoding added
        """
        seq_len = x.shape[1]

        # Add positional encoding
        pos_enc = Tensor(self.pe[:seq_len, :], requires_grad=False, device=x.device)
        x = x + pos_enc

        if self.dropout is not None:
            x = self.dropout(x)

        return x
