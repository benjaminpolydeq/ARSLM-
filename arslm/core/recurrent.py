"""
Adaptive Recurrent Neural Networks for ARSLM.

Implements adaptive RNN, LSTM, and GRU variants that can dynamically
adjust their behavior based on input patterns.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class AdaptiveLSTM(nn.Module):
    """
    Adaptive Long Short-Term Memory network.
    
    Enhanced LSTM with adaptive mechanisms for better sequence modeling.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.1,
        bidirectional: bool = False
    ):
        """
        Initialize Adaptive LSTM.
        
        Args:
            input_size: Size of input features
            hidden_size: Size of hidden state
            num_layers: Number of LSTM layers
            dropout: Dropout probability
            bidirectional: Whether to use bidirectional LSTM
        """
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # Core LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Adaptive gating mechanism
        self.adaptive_gate = nn.Sequential(
            nn.Linear(hidden_size * self.num_directions, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid()
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        input: torch.Tensor,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through Adaptive LSTM.
        
        Args:
            input: Input tensor [batch_size, seq_length, input_size]
            hidden_state: Tuple of (h_0, c_0) hidden states
            
        Returns:
            - Output tensor [batch_size, seq_length, hidden_size * num_directions]
            - Tuple of final (h_n, c_n) states
        """
        batch_size, seq_length, _ = input.size()
        
        # Initialize hidden state if not provided
        if hidden_state is None:
            h_0 = torch.zeros(
                self.num_layers * self.num_directions,
                batch_size,
                self.hidden_size,
                device=input.device,
                dtype=input.dtype
            )
            c_0 = torch.zeros_like(h_0)
            hidden_state = (h_0, c_0)
        
        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(input, hidden_state)
        
        # Apply adaptive gating
        gate = self.adaptive_gate(lstm_out)
        output = lstm_out * gate
        
        output = self.dropout(output)
        
        return output, (h_n, c_n)


class AdaptiveGRU(nn.Module):
    """
    Adaptive Gated Recurrent Unit.
    
    GRU variant with adaptive mechanisms for dynamic sequence processing.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.1,
        bidirectional: bool = False
    ):
        """
        Initialize Adaptive GRU.
        
        Args:
            input_size: Size of input features
            hidden_size: Size of hidden state
            num_layers: Number of GRU layers
            dropout: Dropout probability
            bidirectional: Whether to use bidirectional GRU
        """
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # Core GRU
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Adaptive mechanism
        self.adaptive_layer = nn.Sequential(
            nn.Linear(hidden_size * self.num_directions, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size * self.num_directions)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        input: torch.Tensor,
        hidden_state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through Adaptive GRU.
        
        Args:
            input: Input tensor [batch_size, seq_length, input_size]
            hidden_state: Initial hidden state [num_layers * num_directions, batch_size, hidden_size]
            
        Returns:
            - Output tensor [batch_size, seq_length, hidden_size * num_directions]
            - Final hidden state
        """
        batch_size = input.size(0)
        
        # Initialize hidden state if not provided
        if hidden_state is None:
            hidden_state = torch.zeros(
                self.num_layers * self.num_directions,
                batch_size,
                self.hidden_size,
                device=input.device,
                dtype=input.dtype
            )
        
        # GRU forward pass
        gru_out, h_n = self.gru(input, hidden_state)
        
        # Apply adaptive transformation
        adaptive_out = self.adaptive_layer(gru_out)
        output = gru_out + adaptive_out
        
        output = self.dropout(output)
        
        return output, h_n


class AdaptiveRNN(nn.Module):
    """
    Adaptive Vanilla RNN with enhanced capabilities.
    
    Simple RNN with adaptive gating for better gradient flow.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.1,
        bidirectional: bool = False
    ):
        """
        Initialize Adaptive RNN.
        
        Args:
            input_size: Size of input features
            hidden_size: Size of hidden state
            num_layers: Number of RNN layers
            dropout: Dropout probability
            bidirectional: Whether to use bidirectional RNN
        """
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # Core RNN
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=True,
            nonlinearity='tanh'
        )
        
        # Residual connection
        if input_size == hidden_size * self.num_directions:
            self.use_residual = True
        else:
            self.use_residual = False
            self.residual_proj = nn.Linear(
                input_size,
                hidden_size * self.num_directions
            )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        input: torch.Tensor,
        hidden_state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through Adaptive RNN.
        
        Args:
            input: Input tensor [batch_size, seq_length, input_size]
            hidden_state: Initial hidden state
            
        Returns:
            - Output tensor [batch_size, seq_length, hidden_size * num_directions]
            - Final hidden state
        """
        batch_size = input.size(0)
        
        # Initialize hidden state if not provided
        if hidden_state is None:
            hidden_state = torch.zeros(
                self.num_layers * self.num_directions,
                batch_size,
                self.hidden_size,
                device=input.device,
                dtype=input.dtype
            )
        
        # RNN forward pass
        rnn_out, h_n = self.rnn(input, hidden_state)
        
        # Apply residual connection
        if self.use_residual:
            output = rnn_out + input
        else:
            output = rnn_out + self.residual_proj(input)
        
        output = self.dropout(output)
        
        return output, h_n


class StackedRecurrent(nn.Module):
    """
    Stacked recurrent layers with different types.
    
    Allows combining LSTM, GRU, and RNN in a single architecture.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        layer_types: list = ['lstm', 'gru', 'lstm'],
        dropout: float = 0.1
    ):
        """
        Initialize Stacked Recurrent network.
        
        Args:
            input_size: Size of input features
            hidden_size: Size of hidden state
            layer_types: List of layer types ('lstm', 'gru', 'rnn')
            dropout: Dropout probability
        """
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.layer_types = layer_types
        
        # Create layers
        self.layers = nn.ModuleList()
        current_input_size = input_size
        
        for layer_type in layer_types:
            if layer_type.lower() == 'lstm':
                layer = AdaptiveLSTM(
                    current_input_size,
                    hidden_size,
                    num_layers=1,
                    dropout=dropout
                )
            elif layer_type.lower() == 'gru':
                layer = AdaptiveGRU(
                    current_input_size,
                    hidden_size,
                    num_layers=1,
                    dropout=dropout
                )
            elif layer_type.lower() == 'rnn':
                layer = AdaptiveRNN(
                    current_input_size,
                    hidden_size,
                    num_layers=1,
                    dropout=dropout
                )
            else:
                raise ValueError(f"Unknown layer type: {layer_type}")
            
            self.layers.append(layer)
            current_input_size = hidden_size
        
    def forward(
        self,
        input: torch.Tensor,
        hidden_states: Optional[list] = None
    ) -> Tuple[torch.Tensor, list]:
        """
        Forward pass through stacked layers.
        
        Args:
            input: Input tensor
            hidden_states: List of hidden states for each layer
            
        Returns:
            - Output tensor
            - List of final hidden states
        """
        if hidden_states is None:
            hidden_states = [None] * len(self.layers)
        
        output = input
        new_hidden_states = []
        
        for i, (layer, hidden) in enumerate(zip(self.layers, hidden_states)):
            output, new_hidden = layer(output, hidden)
            new_hidden_states.append(new_hidden)
        
        return output, new_hidden_states