import json, copy
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import average_precision_score
from src import config as C
from src.preprocessing import load_dev  # dev.npz only; final_test is never loaded
from src.evaluation import best_f1_threshold, classification_report
from src.reliability import reliability_stats
from src.models.mlp import MLP

OUT = C.ROOT / "experiments" / "mlp_baseline"
HP = dict(hidden=64, lr=1e-3, batch_size=256, max_epochs=100, patience=10)


def _scores(model, X):
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(X)).numpy()


def make_optimizer(name, params, lr, weight_decay=0.0):
    """Optimizer factory; lr shared, everything else at library/standard defaults (no tuning)."""
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd_momentum":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    if name == "rmsprop":
        return torch.optim.RMSprop(params, lr=lr, weight_decay=weight_decay)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    raise ValueError(name)


def train_mlp(activation="relu", out=OUT, optimizer="adam", weight_decay=0.0, dropout=0.0, lr=None, hidden=None):
    hp = dict(HP, lr=HP["lr"] if lr is None else lr,  # None -> established baseline lr (1e-3)
              hidden=HP["hidden"] if hidden is None else hidden)  # None -> established baseline width (64)
    torch.manual_seed(C.SEED); np.random.seed(C.SEED)
    d = load_dev()
    Xtr, ytr = torch.from_numpy(d["X_sup_train"]), torch.from_numpy(d["y_sup_train"].astype(np.float32))
    Xva, yva = torch.from_numpy(d["X_sup_val"]), d["y_sup_val"]
    pos_weight = torch.tensor((ytr == 0).sum().item() / (ytr == 1).sum().item())
    model = MLP(Xtr.shape[1], hp["hidden"], activation, dropout)
    opt = make_optimizer(optimizer, model.parameters(), hp["lr"], weight_decay)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=HP["batch_size"], shuffle=True)  # tabular rows: shuffle within sup_train only
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
    m.update(activation=activation, optimizer=optimizer, weight_decay=weight_decay, dropout=dropout, best_epoch=int(np.argmax([h["val_pr_auc"] for h in hist]) + 1), epochs_run=len(hist),
             pos_weight=float(pos_weight), hyperparameters=hp, split="sup_val (threshold also selected on sup_val)")
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "n_features": Xtr.shape[1], "hidden": hp["hidden"], "threshold": thr, "activation": activation, "dropout": dropout, "weight_decay": weight_decay}, out / "best_model.pt")
    (out / "history.json").write_text(json.dumps(hist, indent=2))
    (out / "metrics.json").write_text(json.dumps(m, indent=2))
    return m
