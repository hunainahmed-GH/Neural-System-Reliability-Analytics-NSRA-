import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score
from src import config as C
from src.preprocessing import load_dev
from src.reliability import segments


def _save(name):
    plt.tight_layout(); plt.savefig(C.EDA_DIR / name, dpi=110); plt.close()


def run_eda():
    """EDA uses train + dev only. final_test is never read."""
    C.EDA_DIR.mkdir(parents=True, exist_ok=True)
    meta = json.loads((C.PROC_DIR / "meta.json").read_text())
    names = meta["kept_features"]
    d = load_dev()
    tr = np.vstack([d["X_train_fit"], d["X_train_val"]])
    dev = np.vstack([d["X_sup_train"], d["X_sup_val"]])
    y = np.r_[d["y_sup_train"], d["y_sup_val"]]

    # data quality
    dupes = lambda X: int(len(X) - len(np.unique(X, axis=0)))
    summary = {"train_rows": len(tr), "dev_rows": len(dev), "missing_train": int(np.isnan(tr).sum()),
               "missing_dev": int(np.isnan(dev).sum()), "duplicate_rows_train": dupes(tr), "duplicate_rows_dev": dupes(dev),
               "dev_anomaly_rate": float(y.mean()), "dev_anomaly_segments": len(segments(y))}

    # per-feature separation (ROC-AUC on dev) and drift (KS train vs dev-normal)
    auc = np.array([roc_auc_score(y, dev[:, j]) for j in range(dev.shape[1])])
    sep = np.abs(auc - 0.5) * 2
    ks = np.array([ks_2samp(tr[:, j], dev[y == 0, j]).statistic for j in range(dev.shape[1])])
    o = np.argsort(-sep)
    summary["top_separating_features"] = {names[j]: round(float(auc[j]), 3) for j in o[:5]}
    o2 = np.argsort(-ks)
    summary["top_drift_features_ks"] = {names[j]: round(float(ks[j]), 3) for j in o2[:5]}
    summary["mean_ks_train_vs_dev_normal"] = float(ks.mean())
    f33 = names.index("f33") if "f33" in names else None
    if f33 is not None:
        summary["f33_dev_auc"] = round(float(auc[f33]), 3)
    meta["eda_findings"] = summary
    (C.PROC_DIR / "meta.json").write_text(json.dumps(meta, indent=2))
    (C.EDA_DIR / "eda_summary.json").write_text(json.dumps(summary, indent=2))

    # plots
    fig, ax = plt.subplots(2, 1, figsize=(12, 4), sharex=True)
    ax[0].plot(y, lw=.5); ax[0].set_title("dev labels (sup_train + sup_val, chronological)")
    ax[1].plot(dev[:, f33 if f33 is not None else 0], lw=.5); ax[1].set_title("f33 over time")
    ax[0].axvline(len(d["y_sup_train"]), c="r", ls="--"); _save("labels_timeline.png")

    plt.figure(figsize=(10, 4)); plt.bar(names, sep); plt.xticks(rotation=90); plt.title("anomaly separation |2*AUC-1| per feature (dev)"); _save("feature_separation.png")
    plt.figure(figsize=(10, 4)); plt.bar(names, ks, color="C1"); plt.xticks(rotation=90); plt.title("drift: KS(train, dev-normal) per feature"); _save("drift_ks.png")

    fig, ax = plt.subplots(2, 3, figsize=(13, 6))
    for a, j in zip(ax.ravel(), list(o[:3]) + list(o2[:3])):
        a.hist(tr[:, j], 50, alpha=.5, density=True, label="train"); a.hist(dev[y == 0, j], 50, alpha=.5, density=True, label="dev normal")
        a.hist(dev[y == 1, j], 50, alpha=.5, density=True, label="dev anomaly"); a.set_title(names[j]); a.legend(fontsize=6)
    _save("distributions_top_features.png")

    plt.figure(figsize=(7, 6)); plt.imshow(np.corrcoef(tr.T), cmap="coolwarm", vmin=-1, vmax=1); plt.colorbar(); plt.title("train feature correlation"); _save("correlation_train.png")
    fig, ax = plt.subplots(4, 1, figsize=(12, 7), sharex=True)
    for a, j in zip(ax, o[:4]):
        a.plot(dev[:, j], lw=.5)
        for s, e in segments(y): a.axvspan(s, e, color="r", alpha=.2, lw=0)
        a.set_ylabel(names[j])
    _save("temporal_top_features.png")
    return summary
