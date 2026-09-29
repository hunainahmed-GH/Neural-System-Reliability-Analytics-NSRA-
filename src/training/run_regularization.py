import csv, json
from src import config as C
from src.training.train_mlp import train_mlp

OUT = C.ROOT / "experiments" / "regularization"
L2, DROPOUT = 1e-4, 0.20  # fixed, not tuned
CONFIGS = {  # name: (weight_decay, dropout)
    "none": (0.0, 0.0),
    "l2": (L2, 0.0),
    "dropout": (0.0, DROPOUT),
    "l2_dropout": (L2, DROPOUT),
}
KEYS = ["precision", "recall", "f1", "roc_auc", "pr_auc"]


def run_regularization():
    """Same MLP(30->64->ReLU->[Dropout]->1)/Adam/seed/lr/batch/loss/pos_weight/patience; only regularization differs.
    Trains on sup_train, selects on sup_val only; final_test is never loaded."""
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, (wd, dp) in CONFIGS.items():
        m = train_mlp("relu", OUT / name, optimizer="adam", weight_decay=wd, dropout=dp)
        rows.append({"config": name, "weight_decay": wd, "dropout": dp, **{k: round(m[k], 4) for k in KEYS},
                     "threshold": round(m["threshold"], 4), "confusion_matrix": m["confusion_matrix"],
                     "best_epoch": m["best_epoch"], "epochs_run": m["epochs_run"],
                     "false_alarm_rate": round(m["reliability"]["false_alarm_rate"], 4)})
    (OUT / "comparison.json").write_text(json.dumps(rows, indent=2))
    with open(OUT / "comparison.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    return rows
