"""Transformer model architecture.

Implementation of "Attention is All You Need" (Vaswani et al., 2017).

This module provides:
- TransformerEncoderLayer: Single encoder layer with self-attention
- TransformerDecoderLayer: Single decoder layer with cross-attention
- TransformerEncoder: Stack of encoder layers
- TransformerDecoder: Stack of decoder layers
- Transformer: Complete encoder-decoder transformer
- GPTModel: Decoder-only transformer for language modeling
"""

import numpy as np
from numpy_dl.core.module import Module, ModuleList
from numpy_dl.core.tensor import Tensor
from numpy_dl.nn import Linear, Dropout, LayerNorm, Embedding
from numpy_dl.nn.attention import (
    MultiHeadAttention,
    PositionwiseFeedForward,
    PositionalEncoding
)
from numpy_dl.utils.logging import get_logger
from typing import Optional


class TransformerEncoderLayer(Module):
    """
    Single Transformer Encoder Layer.

    Consists of:
    1. Multi-head self-attention
    2. Add & Norm
    3. Position-wise feed-forward
    4. Add & Norm

    Args:
        d_model: Dimension of the model
        num_heads: Number of attention heads
        d_ff: Dimension of feed-forward layer
        dropout: Dropout probability

    Example:
        >>> layer = TransformerEncoderLayer(d_model=512, num_heads=8, d_ff=2048)
        >>> output = layer(x, mask=None)
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1
    ):
        super().__init__()

        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)

        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (batch, seq_len, d_model)
            mask: Optional attention mask

        Returns:
            Output tensor (batch, seq_len, d_model)
        """
        # Self-attention with residual connection
        attn_output, _ = self.self_attn(x, x, x, mask)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)

        # Feed-forward with residual connection
        ffn_output = self.ffn(x)
        x = x + self.dropout2(ffn_output)
        x = self.norm2(x)

        return x


class TransformerDecoderLayer(Module):
    """
    Single Transformer Decoder Layer.

    Consists of:
    1. Masked multi-head self-attention
    2. Add & Norm
    3. Multi-head cross-attention (to encoder output)
    4. Add & Norm
    5. Position-wise feed-forward
    6. Add & Norm

    Args:
        d_model: Dimension of the model
        num_heads: Number of attention heads
        d_ff: Dimension of feed-forward layer
        dropout: Dropout probability

    Example:
        >>> layer = TransformerDecoderLayer(d_model=512, num_heads=8, d_ff=2048)
        >>> output = layer(x, encoder_output, src_mask=None, tgt_mask=None)
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1
    ):
        super().__init__()

        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)

        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        encoder_output: Tensor,
        src_mask: Optional[Tensor] = None,
        tgt_mask: Optional[Tensor] = None
    ) -> Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (batch, tgt_seq_len, d_model)
            encoder_output: Encoder output (batch, src_seq_len, d_model)
            src_mask: Source attention mask
            tgt_mask: Target attention mask (causal mask)

        Returns:
            Output tensor (batch, tgt_seq_len, d_model)
        """
        # Masked self-attention with residual connection
        attn_output, _ = self.self_attn(x, x, x, tgt_mask)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)

        # Cross-attention with residual connection
        attn_output, _ = self.cross_attn(x, encoder_output, encoder_output, src_mask)
        x = x + self.dropout2(attn_output)
        x = self.norm2(x)

        # Feed-forward with residual connection
        ffn_output = self.ffn(x)
        x = x + self.dropout3(ffn_output)
        x = self.norm3(x)

        return x


class TransformerEncoder(Module):
    """
    Transformer Encoder: Stack of encoder layers.

    Args:
        num_layers: Number of encoder layers
        d_model: Dimension of the model
        num_heads: Number of attention heads
        d_ff: Dimension of feed-forward layer
        dropout: Dropout probability

    Example:
        >>> encoder = TransformerEncoder(num_layers=6, d_model=512, num_heads=8, d_ff=2048)
        >>> output = encoder(x, mask=None)
    """

    def __init__(
        self,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1
    ):
        super().__init__()

        self.layers = ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass through all encoder layers.

        Args:
            x: Input tensor (batch, seq_len, d_model)
            mask: Optional attention mask

        Returns:
            Output tensor (batch, seq_len, d_model)
        """
        for layer in self.layers:
            x = layer(x, mask)
        return x


class TransformerDecoder(Module):
    """
    Transformer Decoder: Stack of decoder layers.

    Args:
        num_layers: Number of decoder layers
        d_model: Dimension of the model
        num_heads: Number of attention heads
        d_ff: Dimension of feed-forward layer
        dropout: Dropout probability

    Example:
        >>> decoder = TransformerDecoder(num_layers=6, d_model=512, num_heads=8, d_ff=2048)
        >>> output = decoder(x, encoder_output, src_mask=None, tgt_mask=None)
    """

    def __init__(
        self,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1
    ):
        super().__init__()

        self.layers = ModuleList([
            TransformerDecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

    def forward(
        self,
        x: Tensor,
        encoder_output: Tensor,
        src_mask: Optional[Tensor] = None,
        tgt_mask: Optional[Tensor] = None
    ) -> Tensor:
        """
        Forward pass through all decoder layers.

        Args:
            x: Input tensor (batch, tgt_seq_len, d_model)
            encoder_output: Encoder output (batch, src_seq_len, d_model)
            src_mask: Source attention mask
            tgt_mask: Target attention mask (causal mask)

        Returns:
            Output tensor (batch, tgt_seq_len, d_model)
        """
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return x


class Transformer(Module):
    """
    Complete Transformer model (Encoder-Decoder).

    Args:
        src_vocab_size: Source vocabulary size
        tgt_vocab_size: Target vocabulary size
        d_model: Dimension of the model (default: 512)
        num_heads: Number of attention heads (default: 8)
        num_encoder_layers: Number of encoder layers (default: 6)
        num_decoder_layers: Number of decoder layers (default: 6)
        d_ff: Dimension of feed-forward layer (default: 2048)
        max_seq_len: Maximum sequence length (default: 5000)
        dropout: Dropout probability (default: 0.1)

    Example:
        >>> model = Transformer(src_vocab_size=10000, tgt_vocab_size=10000)
        >>> output = model(src, tgt, src_mask=None, tgt_mask=None)
    """

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 512,
        num_heads: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        d_ff: int = 2048,
        max_seq_len: int = 5000,
        dropout: float = 0.1
    ):
        super().__init__()

        self.logger = get_logger('transformer')

        # Embeddings
        from numpy_dl.nn.embedding import Embedding
        self.src_embedding = Embedding(src_vocab_size, d_model)
        self.tgt_embedding = Embedding(tgt_vocab_size, d_model)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)

        # Encoder and Decoder
        self.encoder = TransformerEncoder(
            num_encoder_layers, d_model, num_heads, d_ff, dropout
        )
        self.decoder = TransformerDecoder(
            num_decoder_layers, d_model, num_heads, d_ff, dropout
        )

        # Output projection
        self.output_projection = Linear(d_model, tgt_vocab_size)

        self.d_model = d_model

    def forward(
        self,
        src: Tensor,
        tgt: Tensor,
        src_mask: Optional[Tensor] = None,
        tgt_mask: Optional[Tensor] = None
    ) -> Tensor:
        """
        Forward pass.

        Args:
            src: Source sequence (batch, src_seq_len)
            tgt: Target sequence (batch, tgt_seq_len)
            src_mask: Source attention mask
            tgt_mask: Target attention mask (causal)

        Returns:
            Output logits (batch, tgt_seq_len, tgt_vocab_size)
        """
        # Embed and add positional encoding
        src_embed = self.src_embedding(src) * np.sqrt(self.d_model)
        tgt_embed = self.tgt_embedding(tgt) * np.sqrt(self.d_model)

        src_embed = self.pos_encoding(src_embed)
        tgt_embed = self.pos_encoding(tgt_embed)

        # Encode source
        encoder_output = self.encoder(src_embed, src_mask)

        # Decode target
        decoder_output = self.decoder(tgt_embed, encoder_output, src_mask, tgt_mask)

        # Project to vocabulary
        output = self.output_projection(decoder_output)

        return output


class GPTModel(Module):
    """
    GPT-style decoder-only transformer for language modeling.

    Uses causal (autoregressive) masking to predict next tokens.

    Args:
        vocab_size: Vocabulary size
        d_model: Dimension of the model (default: 256)
        num_heads: Number of attention heads (default: 4)
        num_layers: Number of transformer layers (default: 4)
        d_ff: Dimension of feed-forward layer (default: 1024)
        max_seq_len: Maximum sequence length (default: 512)
        dropout: Dropout probability (default: 0.1)

    Example:
        >>> model = GPTModel(vocab_size=10000, d_model=256, num_heads=4, num_layers=4)
        >>> output = model(x)  # (batch, seq_len, vocab_size)
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        num_heads: int = 4,
        num_layers: int = 4,
        d_ff: int = 1024,
        max_seq_len: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()

        self.logger = get_logger('transformer')
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # Token embedding
        from numpy_dl.nn.embedding import Embedding
        self.token_embedding = Embedding(vocab_size, d_model)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)

        # Transformer decoder layers (used as autoregressive model)
        self.layers = ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        # Output projection
        self.output_projection = Linear(d_model, vocab_size)

        self.logger.info(
            "Initialized GPT model",
            vocab_size=vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            max_seq_len=max_seq_len
        )

    def _create_causal_mask(self, seq_len: int) -> np.ndarray:
        """
        Create causal mask for autoregressive attention.

        Args:
            seq_len: Sequence length

        Returns:
            Causal mask (seq_len, seq_len)
        """
        # Create upper triangular mask (1s above diagonal, 0s on/below)
        mask = np.triu(np.ones((seq_len, seq_len)), k=1)
        return mask

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.

        Args:
            x: Input token indices (batch, seq_len)

        Returns:
            Output logits (batch, seq_len, vocab_size)
        """
        batch_size, seq_len = x.shape

        # Token embedding with scaling
        x = self.token_embedding(x) * np.sqrt(self.d_model)

        # Add positional encoding
        x = self.pos_encoding(x)

        # Create causal mask
        causal_mask = Tensor(
            self._create_causal_mask(seq_len),
            requires_grad=False,
            device=x.device
        )

        # Pass through transformer layers
        for layer in self.layers:
            x = layer(x, mask=causal_mask)

        # Project to vocabulary
        logits = self.output_projection(x)

        return logits

    def generate(
        self,
        start_tokens: Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0
    ) -> Tensor:
        """
        Generate text autoregressively.

        Args:
            start_tokens: Starting token indices (batch, start_len)
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature (higher = more random)

        Returns:
            Generated token indices (batch, start_len + max_new_tokens)
        """
        self.eval()
        generated = start_tokens

        for _ in range(max_new_tokens):
            # Get predictions for current sequence
            logits = self.forward(generated)  # (batch, seq_len, vocab_size)

            # Get logits for last token
            next_token_logits = logits[:, -1, :]  # (batch, vocab_size)

            # Apply temperature
            next_token_logits = next_token_logits / temperature

            # Convert to probabilities
            probs = next_token_logits.exp()
            probs = probs / probs.sum(axis=-1, keepdims=True)

            # Sample from distribution
            probs_np = probs.numpy()
            next_token = np.array([
                np.random.choice(len(p), p=p) for p in probs_np
            ]).reshape(-1, 1)

            next_token_tensor = Tensor(next_token, device=generated.device)

            # Append to generated sequence
            generated = Tensor(
                np.concatenate([generated.numpy(), next_token_tensor.numpy()], axis=1),
                device=generated.device
            )

            # Stop if we exceed max length
            if generated.shape[1] >= self.max_seq_len:
                break

        return generated
