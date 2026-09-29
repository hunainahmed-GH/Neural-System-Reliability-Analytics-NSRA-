import csv, json
from src import config as C
from src.training.train_mlp import train_mlp

OUT = C.ROOT / "experiments" / "hyperparameters" / "hidden_size"
HIDDEN_SIZES = [16, 32, 64, 128, 256]  # fixed grid, nothing else tuned
KEYS = ["precision", "recall", "f1", "roc_auc", "pr_auc"]


def run_hyperparameter_hidden():
    """Same Adam/lr 1e-3/seed/batch/loss/pos_weight/max_epochs/patience, ReLU, no regularization; only hidden width H differs.
    MLP 30->Dense(H)->ReLU->Dense(1). Trains on sup_train, selects on sup_val only; final_test is never loaded."""
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for h in HIDDEN_SIZES:
        m = train_mlp("relu", OUT / f"h_{h}", optimizer="adam", weight_decay=0.0, dropout=0.0, hidden=h)
        rows.append({"config": f"h_{h}", "hidden": h, **{k: round(m[k], 4) for k in KEYS},
                     "threshold": round(m["threshold"], 4), "confusion_matrix": m["confusion_matrix"],
                     "best_epoch": m["best_epoch"], "epochs_run": m["epochs_run"],
                     "false_alarm_rate": round(m["reliability"]["false_alarm_rate"], 4)})
    (OUT / "comparison.json").write_text(json.dumps(rows, indent=2))
    with open(OUT / "comparison.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    return rows
