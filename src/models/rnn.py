import torch.nn as nn


class SimpleRNN(nn.Module):
    """Vanilla Elman RNN: (batch, seq_len, input_size) -> RNN(hidden, tanh) -> last timestep hidden state -> Dense(1).
    No LSTM/GRU, no bidirectionality, no attention, no dropout (this experiment)."""
    def __init__(self, input_size=30, hidden_size=64, num_layers=1):
        super().__init__()
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                           nonlinearity="tanh", batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        out, _ = self.rnn(x)
        last = out[:, -1, :]  # final timestep hidden state
        return self.fc(last).squeeze(-1)
