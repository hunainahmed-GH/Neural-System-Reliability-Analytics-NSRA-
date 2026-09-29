"""Final ANN comparison: consolidates EXISTING metrics.json / comparison files only.
No training, no inference, no data loading (dev.npz / final_test.npz are never opened)."""
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src import config as C

EXP = C.ROOT / "experiments"
OUT = EXP / "results" / "final_comparison"
PLOTS = OUT / "plots"

MAIN = [("MLP", "mlp_baseline"), ("RNN", "rnn"), ("LSTM", "lstm"), ("CNN", "cnn"), ("Autoencoder", "autoencoder")]

# (group, comparison dir, config-column name, baseline config value)
MLP_GROUPS = [
    ("activation", EXP / "activation", "activation", "relu"),
    ("optimizer", EXP / "optimizers", "optimizer", "adam"),
    ("regularization", EXP / "regularization", "config", "none"),
    ("learning_rate", EXP / "hyperparameters" / "learning_rate", "config", "lr_1e-3"),
    ("hidden_size", EXP / "hyperparameters" / "hidden_size", "config", "h_64"),
]

DESC = {
    "MLP": dict(input="row-wise, 30 features", arch="30 -> Dense(64) -> ReLU -> Dense(1)", supervision="supervised",
                loss="BCE (pos_weight)", threshold_rule="max-F1 on sup_val (labels)"),
    "RNN": dict(input="30x30 window (label = last row)", arch="Elman RNN(30->64, tanh, 1 layer) -> last step -> Dense(1)",
                supervision="supervised", loss="BCE (pos_weight)", threshold_rule="max-F1 on sup_val (labels)"),
    "LSTM": dict(input="30x30 window (label = last row)", arch="LSTM(30->64, 1 layer) -> last step -> Dense(1)",
                 supervision="supervised", loss="BCE (pos_weight)", threshold_rule="max-F1 on sup_val (labels)"),
    "CNN": dict(input="30x30 window (label = last row)",
                arch="Conv1d(30->64,k3) -> ReLU -> Conv1d(64->64,k3) -> ReLU -> GAP -> Dense(1)",
                supervision="supervised", loss="BCE (pos_weight)", threshold_rule="max-F1 on sup_val (labels)"),
    "Autoencoder": dict(input="30x30 window, flattened to 900", arch="FC AE 900-128-32-128-900 (ReLU)",
                        supervision="unsupervised (train_fit, normal only)", loss="MSE reconstruction",
                        threshold_rule="99th percentile of train_val recon. error (no labels)"),
}


def n_params(name, hp):
    """Analytic parameter counts from the model definitions in src/models (no torch needed)."""
    if name == "MLP":
        h = hp["hidden"]; return 30 * h + h + h + 1
    if name == "RNN":
        h = hp["hidden"]; return (h * 30 + h * h + 2 * h) + h + 1
    if name == "LSTM":
        h = hp["hidden"]; return (4 * h * 30 + 4 * h * h + 8 * h) + h + 1
    if name == "CNN":
        c, k = hp["conv_channels"], hp["kernel_size"]; return (30 * c * k + c) + (c * c * k + c) + c + 1
    if name == "Autoencoder":
        h, z, f = hp["hidden"], hp["latent"], 900
        return (f * h + h) + (h * z + z) + (z * h + h) + (h * f + f)
    raise KeyError(name)


def cm_stats(cm):
    (tn, fp), (fn, tp) = cm
    pos, neg = tp + fn, tn + fp
    tpr, tnr = tp / pos, tn / neg
    den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return dict(tn=tn, fp=fp, fn=fn, tp=tp, n_eval=tn + fp + fn + tp, n_pos=pos, n_neg=neg,
                prevalence=pos / (pos + neg), fpr=fp / neg, specificity=tnr,
                balanced_accuracy=(tpr + tnr) / 2, mcc=(tp * tn - fp * fn) / den, errors=fp + fn)


def load(p):
    return json.loads(Path(p).read_text())


def write_csv(path, rows):
    cols = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if v is None else v) for k, v in r.items()})


def architecture_table():
    rows, checks = [], []
    for name, d in MAIN:
        m = load(EXP / d / "metrics.json")
        s = cm_stats(m["confusion_matrix"])
        # consistency: stored P/R/F1 must match the confusion matrix
        p, r = s["tp"] / (s["tp"] + s["fp"]), s["tp"] / (s["tp"] + s["fn"])
        f1 = 2 * p * r / (p + r)
        ok = all(abs(a - b) < 1e-9 for a, b in [(p, m["precision"]), (r, m["recall"]), (f1, m["f1"])])
        rel = m.get("reliability")
        if rel:
            ok = ok and abs(rel["false_alarm_rate"] - s["fpr"]) < 1e-9
        checks.append({"model": name, "cm_matches_stored_prf1_and_far": bool(ok)})
        hp = m["hyperparameters"]
        rows.append({
            "model": name, "input_representation": DESC[name]["input"], "architecture": DESC[name]["arch"],
            "supervision": DESC[name]["supervision"], "loss": DESC[name]["loss"],
            "n_params": n_params(name, hp), "threshold_rule": DESC[name]["threshold_rule"],
            "threshold": m["threshold"], "eval_split": "sup_val",
            "precision": m["precision"], "recall": m["recall"], "f1": m["f1"],
            "roc_auc": m["roc_auc"], "pr_auc": m["pr_auc"],
            "tn": s["tn"], "fp": s["fp"], "fn": s["fn"], "tp": s["tp"], "errors_fp_plus_fn": s["errors"],
            "n_eval": s["n_eval"], "n_anomalies": s["n_pos"], "prevalence": round(s["prevalence"], 4),
            "false_alarm_rate": s["fpr"], "specificity": s["specificity"],
            "balanced_accuracy": s["balanced_accuracy"], "mcc": s["mcc"],
            "segment_recall": rel["segment_recall"] if rel else None,
            "detected_segments": rel["detected_segments"] if rel else None,
            "n_segments": rel["n_segments"] if rel else None,
            "mean_detection_delay": rel["mean_detection_delay"] if rel else None,
            "best_epoch": m["best_epoch"], "epochs_run": m["epochs_run"], "max_epochs": hp["max_epochs"],
            "lr": hp["lr"], "batch_size": hp["batch_size"],
            "labels_used_for_selection": name != "Autoencoder",
            "validation_metrics_optimistic": name != "Autoencoder",
        })
    return rows, checks


def mlp_table():
    rows, checks = [], []
    for group, d, col, base in MLP_GROUPS:
        cmp_rows = list(csv.DictReader(open(d / "comparison.csv")))
        for cr in cmp_rows:
            cfg = cr[col]
            m = load(d / cfg / "metrics.json")
            s = cm_stats(m["confusion_matrix"])
            hp, rel = m["hyperparameters"], m["reliability"]
            # consistency vs comparison.csv (rounded to 4 dp)
            ok = all(abs(float(cr[k]) - round(m[k], 4)) < 1e-9 for k in ("precision", "recall", "f1", "roc_auc", "pr_auc"))
            checks.append({"group": group, "config": cfg, "comparison_csv_matches_metrics_json": bool(ok)})
            rows.append({
                "group": group, "config": cfg, "is_reference_config": cfg == base,
                "activation": m.get("activation", "relu"), "optimizer": m.get("optimizer", "adam"),
                "weight_decay": m.get("weight_decay", 0.0), "dropout": m.get("dropout", 0.0),
                "hidden": hp["hidden"], "lr": hp["lr"], "n_params": n_params("MLP", hp),
                "threshold": m["threshold"], "precision": m["precision"], "recall": m["recall"], "f1": m["f1"],
                "roc_auc": m["roc_auc"], "pr_auc": m["pr_auc"],
                "tn": s["tn"], "fp": s["fp"], "fn": s["fn"], "tp": s["tp"], "errors_fp_plus_fn": s["errors"],
                "false_alarm_rate": rel["false_alarm_rate"], "segment_recall": rel["segment_recall"],
                "mean_detection_delay": rel["mean_detection_delay"],
                "best_epoch": m["best_epoch"], "epochs_run": m["epochs_run"], "max_epochs": hp["max_epochs"],
                "hit_epoch_cap": m["epochs_run"] >= hp["max_epochs"],
            })
    return rows, checks


# ---------------------------------------------------------------- plots
COL = {"MLP": "#4C72B0", "RNN": "#DD8452", "LSTM": "#55A868", "CNN": "#C44E52", "Autoencoder": "#8172B3"}


def plot_architecture(rows):
    PLOTS.mkdir(parents=True, exist_ok=True)
    keys = ["precision", "recall", "f1", "roc_auc", "pr_auc"]
    x = np.arange(len(keys)); w = 0.16
    fig, ax = plt.subplots(figsize=(11, 4.8))
    for i, r in enumerate(rows):
        vals = [r[k] for k in keys]
        b = ax.bar(x + (i - 2) * w, vals, w, label=r["model"], color=COL[r["model"]],
                   hatch="//" if r["model"] == "Autoencoder" else None, edgecolor="white")
        for xi, v in zip(x + (i - 2) * w, vals):
            ax.text(xi, v + 0.004, f"{v:.3f}", ha="center", va="bottom", fontsize=6.5, rotation=90)
    ax.set_xticks(x); ax.set_xticklabels(["Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"])
    ax.set_ylim(0.6, 1.08); ax.set_ylabel("score (y-axis truncated at 0.6)")
    ax.set_title("sup_val metrics by architecture (supervised thresholds tuned on sup_val -> optimistic;\n"
                 "Autoencoder: label-free threshold, hatched)", fontsize=10)
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.07), fontsize=8, frameon=False); ax.grid(axis="y", alpha=.3)
    fig.tight_layout(); fig.savefig(PLOTS / "architecture_metrics.png", dpi=150); plt.close(fig)

    # error counts
    fig, ax = plt.subplots(figsize=(7.5, 4))
    x = np.arange(len(rows)); w = 0.38
    fp = [r["fp"] for r in rows]; fn = [r["fn"] for r in rows]
    ax.bar(x - w / 2, fp, w, label="False positives (normal flagged)", color="#E5A93C")
    ax.bar(x + w / 2, fn, w, label="False negatives (anomaly missed)", color="#6A6A6A")
    for xi, v in zip(x - w / 2, fp): ax.text(xi, v + 4, str(v), ha="center", fontsize=8)
    for xi, v in zip(x + w / 2, fn): ax.text(xi, v + 4, str(v), ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels([r["model"] for r in rows]); ax.set_ylabel("count on sup_val")
    ax.set_title("Error counts at each model's own threshold (sup_val)", fontsize=10)
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=.3)
    fig.tight_layout(); fig.savefig(PLOTS / "architecture_error_counts.png", dpi=150); plt.close(fig)

    # confusion matrices (row-normalised, counts annotated)
    fig, axes = plt.subplots(1, 5, figsize=(15, 3.4))
    for ax, r in zip(axes, rows):
        cm = np.array([[r["tn"], r["fp"]], [r["fn"], r["tp"]]])
        ax.imshow(cm / cm.sum(1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=11,
                        color="white" if cm[i, j] / cm[i].sum() > .5 else "black")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["pred N", "pred A"]); ax.set_yticklabels(["true N", "true A"])
        ax.set_title(f"{r['model']} (n={r['n_eval']})", fontsize=9)
    fig.suptitle("Confusion matrices on sup_val (colour = row-normalised)", fontsize=10)
    fig.tight_layout(); fig.savefig(PLOTS / "architecture_confusion_matrices.png", dpi=150); plt.close(fig)


def plot_mlp(rows):
    groups = [g[0] for g in MLP_GROUPS]
    fig, axes = plt.subplots(1, 5, figsize=(17, 4.4), sharey=True)
    for ax, g in zip(axes, groups):
        rr = [r for r in rows if r["group"] == g]
        x = np.arange(len(rr)); w = 0.38
        ax.bar(x - w / 2, [r["f1"] for r in rr], w, label="F1", color="#4C72B0")
        ax.bar(x + w / 2, [r["pr_auc"] for r in rr], w, label="PR-AUC", color="#55A868")
        for xi, r in zip(x, rr):
            ax.text(xi - w / 2, r["f1"] + .004, f"{r['f1']:.3f}", ha="center", fontsize=6, rotation=90)
            ax.text(xi + w / 2, r["pr_auc"] + .004, f"{r['pr_auc']:.3f}", ha="center", fontsize=6, rotation=90)
        ax.set_xticks(x)
        ax.set_xticklabels([r["config"] + ("*" if r["is_reference_config"] else "") for r in rr], rotation=35, ha="right", fontsize=8)
        ax.set_title(g, fontsize=10); ax.set_ylim(0.8, 1.06); ax.grid(axis="y", alpha=.3)
    axes[0].set_ylabel("score (y-axis truncated at 0.8)")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper right", ncol=2, fontsize=8, frameon=False)
    fig.suptitle("MLP experiments on sup_val, one run per config (* = shared reference: ReLU/Adam/no reg/lr 1e-3/h64). "
                 "Thresholds tuned on sup_val.", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(PLOTS / "mlp_experiments_f1_prauc.png", dpi=150); plt.close(fig)


def run_final_comparison():
    OUT.mkdir(parents=True, exist_ok=True)
    arch, c1 = architecture_table()
    mlp, c2 = mlp_table()

    caveats = [
        "Supervised thresholds (max-F1) were chosen on sup_val and checkpoints early-stopped on sup_val PR-AUC; sup_val classification metrics are therefore optimistic.",
        "Autoencoder: trained on normal-only train_fit; checkpoint (train_val MSE) and threshold (99th pct of train_val error) chosen without labels; sup_val labels used for evaluation only.",
        "MLP is row-wise (n=2643); RNN/LSTM/CNN/Autoencoder use 30x30 windows (n=2614; first 29 rows of sup_val have no full window). Same 1178 anomalies, 29 fewer normals.",
        "sup_val contains only 2 contiguous anomaly segments (mean length 589): window/row metrics are strongly autocorrelated; effective independent evidence is small.",
        "sup_val anomaly prevalence (~45%) is far above final_test prevalence (416/8042 = 5.2%): precision and PR-AUC will not transfer directly.",
        "One run per configuration (seed 42 where recorded); no repeated seeds, so small metric gaps (a few FP/FN) are within plausible run-to-run noise.",
        "No winner is declared. final_test has not been used.",
    ]
    (OUT / "architecture_comparison.json").write_text(json.dumps(
        {"eval_split": "sup_val", "final_test_used": False, "caveats": caveats, "models": arch,
         "consistency_checks": c1}, indent=2))
    write_csv(OUT / "architecture_comparison.csv", arch)
    (OUT / "mlp_experiment_comparison.json").write_text(json.dumps(
        {"eval_split": "sup_val", "final_test_used": False,
         "note": "Each group varies one factor from the shared reference config; reference row repeats in every group.",
         "rows": mlp, "consistency_checks": c2}, indent=2))
    write_csv(OUT / "mlp_experiment_comparison.csv", mlp)
    plot_architecture(arch)
    plot_mlp(mlp)
    return {"architecture_rows": len(arch), "mlp_rows": len(mlp),
            "all_checks_pass": all(all(v for k, v in c.items() if k not in ("model", "group", "config")) for c in c1 + c2),
            "out": str(OUT.relative_to(C.ROOT))}
