import torch.nn as nn


class SimpleCNN(nn.Module):
    """Simple 1D temporal CNN, deliberately minimal (no pooling other than the final global average,
    no dropout, batch norm, attention, residuals, or extra conv layers):
    (B, seq_len=30, features=30) -> transpose to (B, channels=30, length=30) ->
    Conv1d(30->64, k=3, pad=1) -> ReLU -> Conv1d(64->64, k=3, pad=1) -> ReLU ->
    AdaptiveAvgPool1d(1) -> Flatten -> Dense(1)."""
    def __init__(self, input_channels=30, conv_channels=64, kernel_size=3):
        super().__init__()
        pad = kernel_size // 2  # padding=1 for kernel_size=3: keeps temporal length at 30
        self.conv1 = nn.Conv1d(input_channels, conv_channels, kernel_size, padding=pad)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(conv_channels, conv_channels, kernel_size, padding=pad)
        self.relu2 = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(conv_channels, 1)

    def forward(self, x):
        # x: (batch, seq_len, features) -> (batch, channels=features, length=seq_len)
        x = x.permute(0, 2, 1)
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.pool(x).flatten(1)
        return self.fc(x).squeeze(-1)
