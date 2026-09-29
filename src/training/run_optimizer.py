import csv, json
from src import config as C
from src.training.train_mlp import train_mlp

OUT = C.ROOT / "experiments" / "optimizers"
OPTIMIZERS = ["sgd", "sgd_momentum", "rmsprop", "adam"]  # sgd_momentum: momentum=0.9; rmsprop: torch defaults
KEYS = ["precision", "recall", "f1", "roc_auc", "pr_auc"]


def run_optimizer():
    """Same MLP(30->64->ReLU->1)/seed/lr/batch/loss/pos_weight/patience for each optimizer; sup_train/sup_val only."""
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in OPTIMIZERS:
        m = train_mlp("relu", OUT / name, optimizer=name)
        rows.append({"optimizer": name, **{k: round(m[k], 4) for k in KEYS}, "threshold": round(m["threshold"], 4),
                     "confusion_matrix": m["confusion_matrix"], "best_epoch": m["best_epoch"], "epochs_run": m["epochs_run"],
                     "false_alarm_rate": round(m["reliability"]["false_alarm_rate"], 4)})
    (OUT / "comparison.json").write_text(json.dumps(rows, indent=2))
    with open(OUT / "comparison.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    return rows
