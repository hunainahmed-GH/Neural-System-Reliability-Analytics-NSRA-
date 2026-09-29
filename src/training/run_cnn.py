from src.training.train_cnn import train_cnn


def run_cnn():
    """1D CNN experiment: same training protocol as the completed RNN/LSTM experiments (Adam, lr 1e-3,
    batch 256, seed 42, BCEWithLogitsLoss, pos_weight methodology, max_epochs/patience, same sup_train
    / sup_val windows and last-timestep label). Only the architecture changes: a simple 2-layer 1D CNN
    with global average pooling in place of a recurrent cell. Trains on sup_train, selects on sup_val
    only; final_test is never loaded."""
    m = train_cnn()
    return [m]
