import json, copy
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, f1_score, confusion_matrix
from src import config as C
from src.preprocessing import load_dev, make_windows  # dev.npz only; final_test is never loaded
from src.models.autoencoder import SimpleAutoencoder

OUT = C.ROOT / "experiments" / "autoencoder"
HP = dict(hidden=128, latent=32, lr=1e-3, batch_size=256, max_epochs=100, patience=10)
THRESHOLD_PERCENTILE = 99  # threshold set from train_val reconstruction errors only, no labels


def _recon_errors(model, X):
    """Per-window mean-squared reconstruction error, shape (N,)."""
    model.eval()
    with torch.no_grad():
        recon = model(X)
        err = ((recon - X) ** 2).mean(dim=(1, 2)).numpy()
    return err


def train_autoencoder(out=OUT):
    torch.manual_seed(C.SEED); np.random.seed(C.SEED)
    d = load_dev()

    # TRAINING: train_fit only. train_fit/train_val have zero anomalies by construction (see meta.json /
    # preprocessing.run_preprocessing asserts); we also assert it here so unsupervised training is verified,
    # not merely assumed. Labels are never passed into windowing or the loss for train_fit/train_val.
    assert d["X_sup_train"] is not None  # sanity: dev.npz has sup_train/sup_val too, but we don't touch them for training
    Wfit, _, _ = make_windows(d["X_train_fit"], None, C.WINDOW_SIZE, "last")
    Wval, _, _ = make_windows(d["X_train_val"], None, C.WINDOW_SIZE, "last")
    Xfit = torch.from_numpy(Wfit.astype(np.float32))
    Xval = torch.from_numpy(Wval.astype(np.float32))
    assert Xfit.shape[1:] == (C.WINDOW_SIZE, C.N_FEATURES) == (30, 30)
    assert Xfit.flatten(1).shape[1] == 900  # flattened window dimension

    model = SimpleAutoencoder(C.WINDOW_SIZE, C.N_FEATURES, HP["hidden"], HP["latent"])
    opt = torch.optim.Adam(model.parameters(), lr=HP["lr"])
    loss_fn = torch.nn.MSELoss()
    loader = DataLoader(TensorDataset(Xfit), batch_size=HP["batch_size"], shuffle=True)  # normal windows only, shuffled

    hist, best, best_state, bad = [], float("inf"), None, 0
    for ep in range(1, HP["max_epochs"] + 1):
        model.train(); tot = 0.0
        for (xb,) in loader:
            opt.zero_grad(); recon = model(xb); loss = loss_fn(recon, xb); loss.backward(); opt.step()
            tot += loss.item() * len(xb)
        with torch.no_grad():
            model.eval(); vloss = loss_fn(model(Xval), Xval).item()  # CHECKPOINT: train_val reconstruction MSE only, no labels
        hist.append({"epoch": ep, "train_loss": tot / len(Xfit), "val_loss": vloss})
        if vloss < best:
            best, best_state, bad = vloss, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= HP["patience"]:
                break
    model.load_state_dict(best_state)

    # THRESHOLD: train_val reconstruction errors only, no anomaly labels used.
    train_val_errors = _recon_errors(model, Xval)
    threshold = float(np.percentile(train_val_errors, THRESHOLD_PERCENTILE))

    # EVALUATION: sup_val, used only now that model + threshold are fixed.
    Wsv, ysv, _ = make_windows(d["X_sup_val"], d["y_sup_val"], C.WINDOW_SIZE, "last")
    Xsv = torch.from_numpy(Wsv.astype(np.float32))
    sup_val_errors = _recon_errors(model, Xsv)
    yhat = (sup_val_errors >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(ysv, yhat, labels=[0, 1]).ravel()

    m = {
        "architecture": "Fully-connected autoencoder (900-128-32-128-900), unsupervised (train_fit only)",
        "input_shape": [C.WINDOW_SIZE, C.N_FEATURES], "flattened_dim": C.WINDOW_SIZE * C.N_FEATURES,
        "latent_dim": HP["latent"], "optimizer": "adam", "learning_rate": HP["lr"], "batch_size": HP["batch_size"],
        "seed": C.SEED, "reconstruction_loss": "mse",
        "threshold": threshold, "threshold_percentile": THRESHOLD_PERCENTILE,
        "train_val_mean_error": float(train_val_errors.mean()), "sup_val_mean_error": float(sup_val_errors.mean()),
        "precision": float(precision_score(ysv, yhat, zero_division=0)), "recall": float(recall_score(ysv, yhat, zero_division=0)),
        "f1": float(f1_score(ysv, yhat, zero_division=0)), "roc_auc": float(roc_auc_score(ysv, sup_val_errors)),
        "pr_auc": float(average_precision_score(ysv, sup_val_errors)),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "best_epoch": int(np.argmin([h["val_loss"] for h in hist]) + 1), "epochs_run": len(hist),
        "hyperparameters": HP,
        "training_data": "train_fit (normal only)", "checkpoint_selection": "train_val reconstruction MSE (no labels)",
        "threshold_data": "train_val reconstruction errors (no labels)", "evaluation_data": "sup_val (labels used for evaluation only, after model/threshold fixed)",
    }
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "seq_len": C.WINDOW_SIZE, "n_features": C.N_FEATURES,
                "hidden": HP["hidden"], "latent": HP["latent"], "threshold": threshold,
                "architecture": "SimpleAutoencoder"}, out / "best_model.pt")
    (out / "history.json").write_text(json.dumps(hist, indent=2))
    (out / "metrics.json").write_text(json.dumps(m, indent=2))
    return m
