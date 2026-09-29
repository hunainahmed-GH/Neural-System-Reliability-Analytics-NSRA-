"""Usage: python run.py [download|preprocess|eda|test|all|mlp|activation|optimizer|regularization|hyperparameter_lr|hyperparameter_hidden|rnn|lstm|cnn|autoencoder|final_comparison|reliability]"""
import sys
from src import config as C
from src.data_loader import download, load_raw
from src.preprocessing import run_preprocessing, window_test
from src.eda import run_eda


def main(cmd):
    if cmd in ("download", "all"):
        download(); tr, te, y = load_raw()
        print(f"[data] loaded train{tr.shape} test{te.shape} labels{y.shape} anomalies={int(y.sum())}")
    if cmd in ("preprocess", "all"):
        m = run_preprocessing()
        print(f"[preprocess] features {m['n_features_raw']}->{m['n_features']} dropped={m['dropped_constant_features']}")
        for k, v in m["splits"].items():
            print(f"  {k:10s} rows={v['rows']:6d} anomalies={v['anomalies']}")
    if cmd in ("eda", "all"):
        print("[eda]", run_eda())
    if cmd in ("test", "all"):
        print("[window test] PASS", window_test())
    if cmd == "mlp":
        from src.training.train_mlp import train_mlp
        print("[mlp]", train_mlp())
    if cmd == "activation":
        from src.training.run_activation import run_activation
        for r in run_activation(): print("[activation]", r)
    if cmd == "optimizer":
        from src.training.run_optimizer import run_optimizer
        for r in run_optimizer(): print("[optimizer]", r)
    if cmd == "regularization":
        from src.training.run_regularization import run_regularization
        for r in run_regularization(): print("[regularization]", r)
    if cmd == "hyperparameter_lr":
        from src.training.run_hyperparameter_lr import run_hyperparameter_lr
        for r in run_hyperparameter_lr(): print("[hyperparameter_lr]", r)
    if cmd == "hyperparameter_hidden":
        from src.training.run_hyperparameter_hidden import run_hyperparameter_hidden
        for r in run_hyperparameter_hidden(): print("[hyperparameter_hidden]", r)
    if cmd == "rnn":
        from src.training.run_rnn import run_rnn
        for r in run_rnn(): print("[rnn]", r)
    if cmd == "lstm":
        from src.training.run_lstm import run_lstm
        for r in run_lstm(): print("[lstm]", r)
    if cmd == "cnn":
        from src.training.run_cnn import run_cnn
        for r in run_cnn(): print("[cnn]", r)
    if cmd == "autoencoder":
        from src.training.run_autoencoder import run_autoencoder
        for r in run_autoencoder(): print("[autoencoder]", r)
    if cmd == "final_comparison":
        from src.final_comparison import run_final_comparison
        print("[final_comparison]", run_final_comparison())
    if cmd == "reliability":
        from src.reliability_runner import run_reliability
        print("[reliability]", run_reliability())
    if cmd not in ("download", "preprocess", "eda", "test", "all", "mlp", "activation", "optimizer", "regularization", "hyperparameter_lr", "hyperparameter_hidden", "rnn", "lstm", "cnn", "autoencoder", "final_comparison", "reliability"):
        sys.exit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "all")
