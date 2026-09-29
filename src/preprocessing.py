import json
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from src import config as C
from src.data_loader import load_raw


def constant_features(train):
    """Constant-feature detection uses TRAIN ONLY."""
    return [int(i) for i in np.where(train.std(0) == 0)[0]]


def make_windows(X, y=None, window=C.WINDOW_SIZE, label_mode=C.LABEL_MODE, stride=1):
    """Sliding windows over ONE split (never call on concatenated splits).
    Returns W (n_win, window, F), y_win (or None), end indices (relative to X)."""
    assert len(X) >= window and label_mode in ("last", "any")
    W = np.ascontiguousarray(sliding_window_view(X, window, axis=0).transpose(0, 2, 1)[::stride])
    ends = np.arange(window - 1, len(X), stride)
    yw = None
    if y is not None:
        yw = y[ends] if label_mode == "last" else sliding_window_view(y, window)[::stride].max(1)
    return W, yw, ends


def build_splits(train, test, y):
    keep = [i for i in range(C.N_FEATURES_RAW) if i not in constant_features(train)]
    src = {"train": train[:, keep], "test": test[:, keep]}
    out = {}
    for name, (s, a, b) in C.SPLITS.items():
        X = src[s][a:b]
        lab = y[a:b] if s == "test" else np.zeros(b - a, np.uint8)
        out[name] = (X, lab)
    return out, keep


def run_preprocessing():
    train, test, y = load_raw()
    assert not np.isnan(train).any() and not np.isnan(test).any()
    const = constant_features(train)
    assert const == C.CONSTANT_FEATURES, const
    splits, keep = build_splits(train, test, y)
    assert len(keep) == C.N_FEATURES
    # row counts / anomaly counts
    for k, (X, lab) in splits.items():
        assert len(X) == C.EXPECTED_ROWS[k], (k, len(X))
        assert X.shape[1] == C.N_FEATURES
        if k in C.EXPECTED_ANOMALIES:
            assert int(lab.sum()) == C.EXPECTED_ANOMALIES[k], (k, int(lab.sum()))
    # chronology / no leakage: contiguous, disjoint, exhaustive per source
    for s in ("train", "test"):
        r = sorted((a, b) for (ss, a, b) in C.SPLITS.values() if ss == s)
        assert r[0][0] == 0 and r[-1][1] == C.N_TRAIN and all(r[i][1] == r[i + 1][0] for i in range(len(r) - 1))
    assert splits["train_fit"][0].std(0).min() > 0, "kept feature constant in train_fit"
    assert splits["train_fit"][1].sum() == 0 and splits["train_val"][1].sum() == 0
    assert 0.0 <= min(X.min() for X, _ in splits.values()) and max(X.max() for X, _ in splits.values()) <= 1.0
    # save: final_test in its own file, dev file never contains it
    C.PROC_DIR.mkdir(parents=True, exist_ok=True)
    dev = dict(X_train_fit=splits["train_fit"][0], X_train_val=splits["train_val"][0],
               X_sup_train=splits["sup_train"][0], y_sup_train=splits["sup_train"][1],
               X_sup_val=splits["sup_val"][0], y_sup_val=splits["sup_val"][1])
    np.savez_compressed(C.PROC_DIR / "dev.npz", **dev)
    np.savez_compressed(C.PROC_DIR / "final_test.npz", X=splits["final_test"][0], y=splits["final_test"][1])
    meta = {
        "dataset": f"SMD {C.MACHINE} (OmniAnomaly/ServerMachineDataset)",
        "n_features_raw": C.N_FEATURES_RAW, "n_features": len(keep),
        "feature_ids_note": "0-based column indices; names f00..f37",
        "dropped_constant_features": [f"f{i:02d}" for i in const],
        "kept_features": [f"f{i:02d}" for i in keep],
        "scaling": C.SCALING, "scaling_note": "source already min-max scaled to [0,1]; no refit",
        "shuffling": False, "window_size": C.WINDOW_SIZE, "label_mode": C.LABEL_MODE,
        "windows_cross_split_boundaries": False,
        "splits": {k: {"source": C.SPLITS[k][0], "start": C.SPLITS[k][1], "end": C.SPLITS[k][2],
                       "rows": len(splits[k][0]), "anomalies": int(splits[k][1].sum()),
                       "purpose": {"train_fit": "AE training / preprocessing fitting", "train_val": "AE validation / threshold calibration",
                                   "sup_train": "supervised ANN training", "sup_val": "supervised ANN validation",
                                   "final_test": "FINAL evaluation only"}[k]} for k in splits},
    }
    (C.PROC_DIR / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def load_dev():
    return dict(np.load(C.PROC_DIR / "dev.npz"))


def load_final_test():
    """FINAL evaluation only."""
    d = np.load(C.PROC_DIR / "final_test.npz")
    return d["X"], d["y"]


def window_test(window=C.WINDOW_SIZE):
    """Checks shapes, label alignment and that windows stay inside each split."""
    dev = load_dev()
    shapes = {}
    for name, xk, yk in [("sup_train", "X_sup_train", "y_sup_train"), ("sup_val", "X_sup_val", "y_sup_val"),
                         ("train_fit", "X_train_fit", None), ("train_val", "X_train_val", None)]:
        X = dev[xk]; y = dev[yk] if yk else None
        W, yw, ends = make_windows(X, y, window, "last")
        assert W.shape == (len(X) - window + 1, window, C.N_FEATURES)
        assert ends[0] == window - 1 and ends[-1] == len(X) - 1  # inside split
        for i in (0, len(W) // 2, len(W) - 1):
            assert np.array_equal(W[i], X[i:i + window])
            if y is not None:
                assert yw[i] == y[i + window - 1]
        shapes[name] = W.shape
    # final_test isolation: dev file has no final_test keys; shape check uses row count from meta only
    assert not any("final" in k for k in dev)
    n = json.loads((C.PROC_DIR / "meta.json").read_text())["splits"]["final_test"]["rows"]
    Wf, _, _ = make_windows(np.zeros((n, C.N_FEATURES), np.float32), None, window)
    assert Wf.shape == (n - window + 1, window, C.N_FEATURES)
    return {"window_shape": (window, C.N_FEATURES), **shapes, "final_test": Wf.shape}
