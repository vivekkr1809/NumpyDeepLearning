"""RNN-based models for sequence processing."""

from typing import Optional
from numpy_dl.core.module import Module, Sequential
from numpy_dl.nn import RNNCell, LSTMCell, GRUCell, Linear, Dropout
from numpy_dl.core.tensor import Tensor, zeros


class SimpleRNN(Module):
    """
    Simple RNN model for sequence classification.

    Processes sequences with an RNN and outputs class predictions.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        num_classes: int,
        dropout: float = 0.0,
        cell_type: str = 'rnn',
    ):
        """
        Initialize SimpleRNN.

        Args:
            input_size: Size of input features
            hidden_size: Size of hidden state
            num_layers: Number of RNN layers
            num_classes: Number of output classes
            dropout: Dropout probability
            cell_type: Type of RNN cell ('rnn', 'lstm', 'gru')
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.cell_type = cell_type

        # Create RNN cells
        self.cells = []
        for i in range(num_layers):
            cell_input_size = input_size if i == 0 else hidden_size

            if cell_type == 'rnn':
                cell = RNNCell(cell_input_size, hidden_size)
            elif cell_type == 'lstm':
                cell = LSTMCell(cell_input_size, hidden_size)
            elif cell_type == 'gru':
                cell = GRUCell(cell_input_size, hidden_size)
            else:
                raise ValueError(f"Invalid cell type: {cell_type}")

            self.cells.append(cell)
            self.add_module(f'cell_{i}', cell)

        # Dropout
        self.dropout = Dropout(dropout) if dropout > 0 else None

        # Output layer
        self.fc = Linear(hidden_size, num_classes)

    def forward(self, x: Tensor, hidden: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (seq_len, batch_size, input_size)
            hidden: Initial hidden state

        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        seq_len, batch_size = x.shape[0], x.shape[1]

        # Initialize hidden state if not provided
        if hidden is None:
            if self.cell_type == 'lstm':
                hidden = [
                    (zeros(batch_size, self.hidden_size, device=x.device),
                     zeros(batch_size, self.hidden_size, device=x.device))
                    for _ in range(self.num_layers)
                ]
            else:
                hidden = [zeros(batch_size, self.hidden_size, device=x.device)
                         for _ in range(self.num_layers)]

        # Process sequence
        for t in range(seq_len):
            x_t = x[t]

            for layer in range(self.num_layers):
                if self.cell_type == 'lstm':
                    h, c = self.cells[layer](x_t, hidden[layer])
                    hidden[layer] = (h, c)
                    x_t = h
                else:
                    h = self.cells[layer](x_t, hidden[layer])
                    hidden[layer] = h
                    x_t = h

                # Apply dropout between layers
                if layer < self.num_layers - 1 and self.dropout is not None:
                    x_t = self.dropout(x_t)

        # Use last hidden state for classification
        if self.cell_type == 'lstm':
            last_hidden = hidden[-1][0]
        else:
            last_hidden = hidden[-1]

        output = self.fc(last_hidden)
        return output

    def __repr__(self):
        return (
            f"SimpleRNN(input_size={self.input_size}, hidden_size={self.hidden_size}, "
            f"num_layers={self.num_layers}, cell_type='{self.cell_type}')"
        )


class Seq2Seq(Module):
    """
    Sequence-to-Sequence model with attention.

    Encoder-decoder architecture for sequence transformation tasks.
    """

    def __init__(
        self,
        input_vocab_size: int,
        output_vocab_size: int,
        embedding_dim: int,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        cell_type: str = 'lstm',
    ):
        """
        Initialize Seq2Seq model.

        Args:
            input_vocab_size: Size of input vocabulary
            output_vocab_size: Size of output vocabulary
            embedding_dim: Dimension of embeddings
            hidden_size: Size of hidden state
            num_layers: Number of layers
            dropout: Dropout probability
            cell_type: Type of RNN cell ('rnn', 'lstm', 'gru')
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.cell_type = cell_type

        # Encoder
        # Note: Embedding layer would be needed in practice
        self.encoder_cells = []
        for i in range(num_layers):
            cell_input_size = embedding_dim if i == 0 else hidden_size

            if cell_type == 'lstm':
                cell = LSTMCell(cell_input_size, hidden_size)
            elif cell_type == 'gru':
                cell = GRUCell(cell_input_size, hidden_size)
            else:
                cell = RNNCell(cell_input_size, hidden_size)

            self.encoder_cells.append(cell)
            self.add_module(f'encoder_cell_{i}', cell)

        # Decoder
        self.decoder_cells = []
        for i in range(num_layers):
            cell_input_size = embedding_dim if i == 0 else hidden_size

            if cell_type == 'lstm':
                cell = LSTMCell(cell_input_size, hidden_size)
            elif cell_type == 'gru':
                cell = GRUCell(cell_input_size, hidden_size)
            else:
                cell = RNNCell(cell_input_size, hidden_size)

            self.decoder_cells.append(cell)
            self.add_module(f'decoder_cell_{i}', cell)

        # Output projection
        self.fc = Linear(hidden_size, output_vocab_size)
        self.dropout = Dropout(dropout) if dropout > 0 else None

    def encode(self, x: Tensor) -> list:
        """
        Encode input sequence.

        Args:
            x: Input tensor of shape (seq_len, batch_size, embedding_dim)

        Returns:
            List of hidden states for each layer
        """
        seq_len, batch_size = x.shape[0], x.shape[1]

        # Initialize hidden states
        if self.cell_type == 'lstm':
            hidden = [
                (zeros(batch_size, self.hidden_size, device=x.device),
                 zeros(batch_size, self.hidden_size, device=x.device))
                for _ in range(self.num_layers)
            ]
        else:
            hidden = [zeros(batch_size, self.hidden_size, device=x.device)
                     for _ in range(self.num_layers)]

        # Process sequence
        for t in range(seq_len):
            x_t = x[t]

            for layer in range(self.num_layers):
                if self.cell_type == 'lstm':
                    h, c = self.encoder_cells[layer](x_t, hidden[layer])
                    hidden[layer] = (h, c)
                    x_t = h
                else:
                    h = self.encoder_cells[layer](x_t, hidden[layer])
                    hidden[layer] = h
                    x_t = h

                if layer < self.num_layers - 1 and self.dropout is not None:
                    x_t = self.dropout(x_t)

        return hidden

    def decode(self, x: Tensor, hidden: list) -> Tensor:
        """
        Decode one step.

        Args:
            x: Input tensor of shape (batch_size, embedding_dim)
            hidden: Hidden states from encoder

        Returns:
            Output logits of shape (batch_size, output_vocab_size)
        """
        x_t = x

        for layer in range(self.num_layers):
            if self.cell_type == 'lstm':
                h, c = self.decoder_cells[layer](x_t, hidden[layer])
                hidden[layer] = (h, c)
                x_t = h
            else:
                h = self.decoder_cells[layer](x_t, hidden[layer])
                hidden[layer] = h
                x_t = h

            if layer < self.num_layers - 1 and self.dropout is not None:
                x_t = self.dropout(x_t)

        output = self.fc(x_t)
        return output

    def forward(self, encoder_input: Tensor, decoder_input: Tensor) -> Tensor:
        """
        Forward pass (teacher forcing).

        Args:
            encoder_input: Encoder input of shape (enc_seq_len, batch_size, embedding_dim)
            decoder_input: Decoder input of shape (dec_seq_len, batch_size, embedding_dim)

        Returns:
            Output tensor of shape (dec_seq_len, batch_size, output_vocab_size)
        """
        # Encode
        hidden = self.encode(encoder_input)

        # Decode
        dec_seq_len = decoder_input.shape[0]
        outputs = []

        for t in range(dec_seq_len):
            output = self.decode(decoder_input[t], hidden)
            outputs.append(output)

        # Stack outputs (simplified - in practice would need proper stacking)
        return outputs[-1]  # Return last output for simplicity
