import csv, json
from src import config as C
from src.training.train_mlp import train_mlp

OUT = C.ROOT / "experiments" / "hyperparameters" / "learning_rate"
LEARNING_RATES = {"lr_1e-4": 1e-4, "lr_3e-4": 3e-4, "lr_1e-3": 1e-3, "lr_3e-3": 3e-3, "lr_1e-2": 1e-2}  # fixed grid, nothing else tuned
KEYS = ["precision", "recall", "f1", "roc_auc", "pr_auc"]


def run_hyperparameter_lr():
    """Same MLP(30->64->ReLU->1)/Adam/seed/batch/loss/pos_weight/max_epochs/patience, no regularization; only lr differs.
    Trains on sup_train, selects on sup_val only; final_test is never loaded."""
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, lr in LEARNING_RATES.items():
        m = train_mlp("relu", OUT / name, optimizer="adam", weight_decay=0.0, dropout=0.0, lr=lr)
        rows.append({"config": name, "lr": lr, **{k: round(m[k], 4) for k in KEYS},
                     "threshold": round(m["threshold"], 4), "confusion_matrix": m["confusion_matrix"],
                     "best_epoch": m["best_epoch"], "epochs_run": m["epochs_run"],
                     "false_alarm_rate": round(m["reliability"]["false_alarm_rate"], 4)})
    (OUT / "comparison.json").write_text(json.dumps(rows, indent=2))
    with open(OUT / "comparison.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    return rows
