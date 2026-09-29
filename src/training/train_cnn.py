import json, copy
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import average_precision_score
from src import config as C
from src.preprocessing import load_dev, make_windows  # dev.npz only; final_test is never loaded
from src.evaluation import best_f1_threshold, classification_report
from src.reliability import reliability_stats
from src.models.cnn import SimpleCNN

OUT = C.ROOT / "experiments" / "cnn"
# Same training protocol as the completed RNN/LSTM experiments; only the architecture changes (1D CNN).
HP = dict(conv_channels=64, kernel_size=3, lr=1e-3, batch_size=256, max_epochs=100, patience=10)


def _scores(model, X):
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(X)).numpy()


def train_cnn(out=OUT):
    torch.manual_seed(C.SEED); np.random.seed(C.SEED)
    d = load_dev()
    # Build actual (N, 30, 30) sliding windows over sup_train / sup_val; labels are last-step labels.
    Wtr, ytr_np, _ = make_windows(d["X_sup_train"], d["y_sup_train"], C.WINDOW_SIZE, "last")
    Wva, yva, _ = make_windows(d["X_sup_val"], d["y_sup_val"], C.WINDOW_SIZE, "last")
    Xtr = torch.from_numpy(Wtr.astype(np.float32))
    ytr = torch.from_numpy(ytr_np.astype(np.float32))
    Xva = torch.from_numpy(Wva.astype(np.float32))
    assert Xtr.shape[1:] == (C.WINDOW_SIZE, C.N_FEATURES) == (30, 30)  # explicit (batch, 30, 30) raw window check

    pos_weight = torch.tensor((ytr == 0).sum().item() / (ytr == 1).sum().item())
    model = SimpleCNN(input_channels=C.N_FEATURES, conv_channels=HP["conv_channels"], kernel_size=HP["kernel_size"])
    opt = torch.optim.Adam(model.parameters(), lr=HP["lr"])
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=HP["batch_size"], shuffle=True)  # shuffle within sup_train windows only

    first_batch_xb, _ = next(iter(loader))
    assert first_batch_xb.shape[1:] == (30, 30)  # explicit first-batch raw input shape check
    with torch.no_grad():
        conv1_out = model.relu1(model.conv1(first_batch_xb.permute(0, 2, 1)))
    assert conv1_out.shape[1] == HP["conv_channels"] and conv1_out.shape[2] == C.WINDOW_SIZE  # (batch, 30 channels->64, length=30) then first conv -> 64 channels, length still 30

    hist, best, best_state, bad = [], -1.0, None, 0
    for ep in range(1, HP["max_epochs"] + 1):
        model.train(); tot = 0.0
        for xb, yb in loader:
            opt.zero_grad(); loss = loss_fn(model(xb), yb); loss.backward(); opt.step()
            tot += loss.item() * len(xb)
        with torch.no_grad():
            model.eval(); vloss = loss_fn(model(Xva), torch.from_numpy(yva.astype(np.float32))).item()
        vpr = average_precision_score(yva, _scores(model, Xva))
        hist.append({"epoch": ep, "train_loss": tot / len(Xtr), "val_loss": vloss, "val_pr_auc": vpr})
        if vpr > best:
            best, best_state, bad = vpr, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= HP["patience"]:
                break
    model.load_state_dict(best_state)
    s = _scores(model, Xva)
    thr = best_f1_threshold(yva, s)  # threshold chosen on sup_val only
    m = classification_report(yva, s, thr)
    m["reliability"] = reliability_stats(yva, (s >= thr).astype(int))
    m.update(architecture="1D CNN (2 conv layers, global average pool, no pooling/dropout/batchnorm)",
              input_shape=[C.WINDOW_SIZE, C.N_FEATURES], sequence_length=C.WINDOW_SIZE, input_channels=C.N_FEATURES,
              conv_channels=HP["conv_channels"], kernel_size=HP["kernel_size"], optimizer="adam", learning_rate=HP["lr"],
              batch_size=HP["batch_size"], seed=C.SEED, pos_weight=float(pos_weight),
              best_epoch=int(np.argmax([h["val_pr_auc"] for h in hist]) + 1), epochs_run=len(hist),
              hyperparameters=HP, split="sup_val (threshold also selected on sup_val)")
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "input_channels": C.N_FEATURES, "conv_channels": HP["conv_channels"],
                "kernel_size": HP["kernel_size"], "threshold": thr, "architecture": "SimpleCNN"}, out / "best_model.pt")
    (out / "history.json").write_text(json.dumps(hist, indent=2))
    (out / "metrics.json").write_text(json.dumps(m, indent=2))
    return m
