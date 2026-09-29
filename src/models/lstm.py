import torch.nn as nn


class SimpleLSTM(nn.Module):
    """Standard LSTM: (batch, seq_len, input_size) -> LSTM(hidden, num_layers) -> last timestep hidden state -> Dense(1).
    Not bidirectional, not stacked, no attention, no dropout, no extra dense layers.
    Only difference from SimpleRNN (src/models/rnn.py): the recurrent cell is LSTM instead of Elman RNN."""
    def __init__(self, input_size=30, hidden_size=64, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                             batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        out, _ = self.lstm(x)
        last = out[:, -1, :]  # final timestep hidden state
        return self.fc(last).squeeze(-1)
