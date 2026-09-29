from src.training.train_lstm import train_lstm


def run_lstm():
    """LSTM experiment: same baseline training setup as the completed RNN experiment (Adam, lr 1e-3,
    batch 256, seed 42, BCEWithLogitsLoss, pos_weight methodology, max_epochs/patience, same sequence
    length and last-timestep-hidden-state -> Dense(1) design). The only model change is RNN -> LSTM.
    Trains on sup_train, selects on sup_val only; final_test is never loaded."""
    m = train_lstm()
    return [m]
