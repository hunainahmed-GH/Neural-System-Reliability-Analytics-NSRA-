from src.training.train_autoencoder import train_autoencoder


def run_autoencoder():
    """Unsupervised autoencoder experiment: trains ONLY on normal train_fit windows (no labels used
    in the loss), checkpoints on train_val reconstruction MSE only, and sets the anomaly threshold as
    the 99th percentile of train_val reconstruction errors (no labels). sup_val labels are used only
    afterward, for evaluation/reporting once the model and threshold are fixed. final_test is never
    loaded."""
    m = train_autoencoder()
    return [m]
