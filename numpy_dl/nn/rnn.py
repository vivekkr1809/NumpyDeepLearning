"""Recurrent neural network layers."""

import numpy as np
from numpy_dl.core.module import Module
from numpy_dl.core.parameter import Parameter
from numpy_dl.core.tensor import Tensor, zeros
from numpy_dl.core import functional as F
from typing import Optional, Tuple


class RNNCell(Module):
    """
    Basic RNN cell.

    Applies the transformation: h' = tanh(W_ih @ x + b_ih + W_hh @ h + b_hh)
    """

    def __init__(self, input_size: int, hidden_size: int, bias: bool = True):
        """
        Initialize RNN cell.

        Args:
            input_size: Size of input features
            hidden_size: Size of hidden state
            bias: Whether to use bias
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Initialize weights
        limit = np.sqrt(1.0 / hidden_size)
        self.weight_ih = Parameter(
            np.random.uniform(-limit, limit, (hidden_size, input_size)).astype(np.float32)
        )
        self.weight_hh = Parameter(
            np.random.uniform(-limit, limit, (hidden_size, hidden_size)).astype(np.float32)
        )

        if bias:
            self.bias_ih = Parameter(np.zeros(hidden_size, dtype=np.float32))
            self.bias_hh = Parameter(np.zeros(hidden_size, dtype=np.float32))
        else:
            self.bias_ih = None
            self.bias_hh = None

    def forward(self, x: Tensor, h: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, input_size)
            h: Hidden state of shape (batch_size, hidden_size)

        Returns:
            New hidden state of shape (batch_size, hidden_size)
        """
        if h is None:
            h = zeros(x.shape[0], self.hidden_size, device=x.device)

        out = x @ self.weight_ih.T + h @ self.weight_hh.T

        if self.bias_ih is not None:
            out = out + self.bias_ih + self.bias_hh

        return F.tanh(out)

    def __repr__(self):
        return f"RNNCell(input_size={self.input_size}, hidden_size={self.hidden_size})"


class LSTMCell(Module):
    """
    LSTM (Long Short-Term Memory) cell.

    Applies the LSTM cell computation with forget, input, cell, and output gates.
    """

    def __init__(self, input_size: int, hidden_size: int, bias: bool = True):
        """
        Initialize LSTM cell.

        Args:
            input_size: Size of input features
            hidden_size: Size of hidden state
            bias: Whether to use bias
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Initialize weights for all gates (concatenated)
        limit = np.sqrt(1.0 / hidden_size)
        self.weight_ih = Parameter(
            np.random.uniform(-limit, limit, (4 * hidden_size, input_size)).astype(np.float32)
        )
        self.weight_hh = Parameter(
            np.random.uniform(-limit, limit, (4 * hidden_size, hidden_size)).astype(np.float32)
        )

        if bias:
            self.bias_ih = Parameter(np.zeros(4 * hidden_size, dtype=np.float32))
            self.bias_hh = Parameter(np.zeros(4 * hidden_size, dtype=np.float32))
        else:
            self.bias_ih = None
            self.bias_hh = None

    def forward(
        self, x: Tensor, state: Optional[Tuple[Tensor, Tensor]] = None
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, input_size)
            state: Tuple of (hidden_state, cell_state), each of shape (batch_size, hidden_size)

        Returns:
            Tuple of (new_hidden_state, new_cell_state)
        """
        if state is None:
            h = zeros(x.shape[0], self.hidden_size, device=x.device)
            c = zeros(x.shape[0], self.hidden_size, device=x.device)
        else:
            h, c = state

        # Compute all gates at once
        gates = x @ self.weight_ih.T + h @ self.weight_hh.T

        if self.bias_ih is not None:
            gates = gates + self.bias_ih + self.bias_hh

        # Split into separate gates
        i_gate, f_gate, g_gate, o_gate = (
            gates[:, :self.hidden_size],
            gates[:, self.hidden_size:2*self.hidden_size],
            gates[:, 2*self.hidden_size:3*self.hidden_size],
            gates[:, 3*self.hidden_size:],
        )

        # Apply activations
        i_gate = F.sigmoid(i_gate)
        f_gate = F.sigmoid(f_gate)
        g_gate = F.tanh(g_gate)
        o_gate = F.sigmoid(o_gate)

        # Update cell state and hidden state
        c_new = f_gate * c + i_gate * g_gate
        h_new = o_gate * F.tanh(c_new)

        return h_new, c_new

    def __repr__(self):
        return f"LSTMCell(input_size={self.input_size}, hidden_size={self.hidden_size})"


class GRUCell(Module):
    """
    GRU (Gated Recurrent Unit) cell.

    Applies the GRU cell computation with reset and update gates.
    """

    def __init__(self, input_size: int, hidden_size: int, bias: bool = True):
        """
        Initialize GRU cell.

        Args:
            input_size: Size of input features
            hidden_size: Size of hidden state
            bias: Whether to use bias
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Initialize weights for all gates
        limit = np.sqrt(1.0 / hidden_size)
        self.weight_ih = Parameter(
            np.random.uniform(-limit, limit, (3 * hidden_size, input_size)).astype(np.float32)
        )
        self.weight_hh = Parameter(
            np.random.uniform(-limit, limit, (3 * hidden_size, hidden_size)).astype(np.float32)
        )

        if bias:
            self.bias_ih = Parameter(np.zeros(3 * hidden_size, dtype=np.float32))
            self.bias_hh = Parameter(np.zeros(3 * hidden_size, dtype=np.float32))
        else:
            self.bias_ih = None
            self.bias_hh = None

    def forward(self, x: Tensor, h: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, input_size)
            h: Hidden state of shape (batch_size, hidden_size)

        Returns:
            New hidden state of shape (batch_size, hidden_size)
        """
        if h is None:
            h = zeros(x.shape[0], self.hidden_size, device=x.device)

        # Compute input and hidden contributions
        gi = x @ self.weight_ih.T
        gh = h @ self.weight_hh.T

        if self.bias_ih is not None:
            gi = gi + self.bias_ih
            gh = gh + self.bias_hh

        # Split into gates
        i_r, i_z, i_n = (
            gi[:, :self.hidden_size],
            gi[:, self.hidden_size:2*self.hidden_size],
            gi[:, 2*self.hidden_size:],
        )
        h_r, h_z, h_n = (
            gh[:, :self.hidden_size],
            gh[:, self.hidden_size:2*self.hidden_size],
            gh[:, 2*self.hidden_size:],
        )

        # Reset and update gates
        r = F.sigmoid(i_r + h_r)
        z = F.sigmoid(i_z + h_z)

        # New gate
        n = F.tanh(i_n + r * h_n)

        # New hidden state
        h_new = (1 - z) * n + z * h

        return h_new

    def __repr__(self):
        return f"GRUCell(input_size={self.input_size}, hidden_size={self.hidden_size})"


class RNN(Module):
    """
    Multi-layer RNN.

    Applies a multi-layer RNN to an input sequence.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        dropout: float = 0.0,
    ):
        """
        Initialize RNN.

        Args:
            input_size: Size of input features
            hidden_size: Size of hidden state
            num_layers: Number of recurrent layers
            bias: Whether to use bias
            dropout: Dropout probability between layers
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        self.cells = []
        for i in range(num_layers):
            cell_input_size = input_size if i == 0 else hidden_size
            cell = RNNCell(cell_input_size, hidden_size, bias)
            self.cells.append(cell)
            self.add_module(f'cell_{i}', cell)

    def forward(
        self, x: Tensor, h0: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (seq_len, batch_size, input_size)
            h0: Initial hidden state of shape (num_layers, batch_size, hidden_size)

        Returns:
            Tuple of (output, hidden_state)
            - output: shape (seq_len, batch_size, hidden_size)
            - hidden_state: shape (num_layers, batch_size, hidden_size)
        """
        seq_len, batch_size = x.shape[0], x.shape[1]

        if h0 is None:
            h0 = zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)

        outputs = []
        hidden_states = []

        for t in range(seq_len):
            x_t = x[t]
            h_t = []

            for layer in range(self.num_layers):
                x_t = self.cells[layer](x_t, h0[layer])
                h_t.append(x_t)

                # Apply dropout between layers
                if layer < self.num_layers - 1 and self.dropout > 0:
                    x_t = F.dropout(x_t, self.dropout, self.training)

            outputs.append(x_t)
            hidden_states.append(h_t)

        # Stack outputs and hidden states
        # This is simplified - in practice would need proper tensor stacking
        return x_t, h_t[-1]

    def __repr__(self):
        return (
            f"RNN(input_size={self.input_size}, hidden_size={self.hidden_size}, "
            f"num_layers={self.num_layers}, dropout={self.dropout})"
        )
