from src.training.train_rnn import train_rnn


def run_rnn():
    """Vanilla Elman RNN experiment: same baseline training setup as the MLP (Adam, lr 1e-3, batch 256,
    seed 42, BCEWithLogitsLoss, pos_weight methodology, max_epochs/patience), the only methodological
    change is row-wise MLP -> 30-step sequence RNN. Trains on sup_train, selects on sup_val only;
    final_test is never loaded."""
    m = train_rnn()
    return [m]
