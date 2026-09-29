import torch.nn as nn


class SimpleAutoencoder(nn.Module):
    """Simple fully-connected sequence autoencoder, applied to a whole flattened window:
    (B, seq_len=30, features=30) -> Flatten(900) -> Dense(128) -> ReLU -> Dense(32) -> ReLU ->
    Dense(128) -> ReLU -> Dense(900) -> reshape back to (B, 30, 30).
    No dropout, batch norm, convolution, recurrence, or attention."""
    def __init__(self, seq_len=30, n_features=30, hidden=128, latent=32):
        super().__init__()
        self.seq_len, self.n_features = seq_len, n_features
        flat = seq_len * n_features
        self.flatten_dim = flat
        self.encoder = nn.Sequential(nn.Flatten(), nn.Linear(flat, hidden), nn.ReLU(),
                                      nn.Linear(hidden, latent), nn.ReLU())
        self.decoder = nn.Sequential(nn.Linear(latent, hidden), nn.ReLU(), nn.Linear(hidden, flat))

    def forward(self, x):
        # x: (batch, seq_len, features)
        z = self.encoder(x)
        out = self.decoder(z)
        return out.view(-1, self.seq_len, self.n_features)
