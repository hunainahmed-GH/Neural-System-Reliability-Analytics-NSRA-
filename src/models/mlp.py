import torch.nn as nn

ACTIVATIONS = {"relu": nn.ReLU, "leakyrelu": nn.LeakyReLU, "tanh": nn.Tanh}


class MLP(nn.Module):
    """n_features -> Dense(hidden) -> activation -> Dense(1); sigmoid is applied via BCEWithLogitsLoss / predict_proba."""
    def __init__(self, n_features=30, hidden=64, activation="relu", dropout=0.0):
        super().__init__()
        layers = [nn.Linear(n_features, hidden), ACTIVATIONS[activation]()]
        if dropout > 0:  # dropout=0 keeps the original layer layout (old checkpoints still load)
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)
