"""Evaluate all saved NSRA checkpoints once on the held-out final-test split.

This is inference-only: it never trains, tunes thresholds, or overwrites existing
experiment artifacts. Results are written to a new final_test_evaluation folder.
"""
from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from src import config as C
from src.models.autoencoder import SimpleAutoencoder
from src.models.cnn import SimpleCNN
from src.models.lstm import SimpleLSTM
from src.models.mlp import MLP
from src.models.rnn import SimpleRNN
from src.preprocessing import make_windows
from src.reliability import ReliabilityEngine


RESULTS_DIR = C.ROOT / "experiments" / "results" / "final_test_evaluation"
MANIFEST_PATH = C.ROOT / "experiments" / "results" / "final_model_freeze" / "freeze_manifest.json"
DATA_PATH = C.PROC_DIR / "final_test.npz"
META_PATH = C.PROC_DIR / "meta.json"


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8-sig") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path.relative_to(C.ROOT)}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_saved_artifacts(candidates: list[dict]) -> None:
    """Refuse evaluation if a checkpoint or threshold-source metrics file drifted."""
    for candidate in candidates:
        for kind in ("checkpoint", "metrics"):
            artifact = candidate.get(kind, {})
            path_value = artifact.get("path")
            expected_hash = artifact.get("sha256")
            if not path_value or not expected_hash:
                raise ValueError(f"Freeze manifest lacks {kind} fingerprint for {candidate.get('model', 'candidate')}")
            path = (C.ROOT / path_value).resolve()
            if not path.is_relative_to(C.ROOT.resolve()) or not path.is_file():
                raise FileNotFoundError(f"Frozen {kind} artifact is missing or outside the project: {path_value}")
            if _sha256(path).lower() != str(expected_hash).lower():
                raise ValueError(f"Frozen {kind} artifact fingerprint mismatch: {path_value}")


def _load_checkpoint(path: Path) -> dict:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # compatibility with older PyTorch releases
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or "state_dict" not in payload:
        raise ValueError(f"Unexpected checkpoint structure in {path.relative_to(C.ROOT)}")
    return payload


def _make_model(name: str, payload: dict, metrics: dict, n_features: int, window_size: int):
    hp = metrics.get("hyperparameters", {})
    if name == "MLP":
        return MLP(
            n_features=int(payload.get("n_features", n_features)),
            hidden=int(payload.get("hidden", hp.get("hidden", 64))),
            activation=str(payload.get("activation", metrics.get("activation", "relu"))),
            dropout=float(payload.get("dropout", metrics.get("dropout", 0.0))),
        ), "row-wise"
    if name == "RNN":
        return SimpleRNN(
            input_size=int(payload.get("input_size", n_features)),
            hidden_size=int(payload.get("hidden_size", metrics.get("hidden_size", 64))),
            num_layers=int(payload.get("num_layers", metrics.get("num_layers", 1))),
        ), "sequence"
    if name == "LSTM":
        return SimpleLSTM(
            input_size=int(payload.get("input_size", n_features)),
            hidden_size=int(payload.get("hidden_size", metrics.get("hidden_size", 64))),
            num_layers=int(payload.get("num_layers", metrics.get("num_layers", 1))),
        ), "sequence"
    if name == "CNN":
        return SimpleCNN(
            input_channels=int(payload.get("input_channels", n_features)),
            conv_channels=int(payload.get("conv_channels", metrics.get("conv_channels", 64))),
            kernel_size=int(payload.get("kernel_size", metrics.get("kernel_size", 3))),
        ), "sequence"
    if name == "Autoencoder":
        return SimpleAutoencoder(
            seq_len=int(payload.get("seq_len", window_size)),
            n_features=int(payload.get("n_features", n_features)),
            hidden=int(payload.get("hidden", hp.get("hidden", 128))),
            latent=int(payload.get("latent", hp.get("latent", 32))),
        ), "autoencoder"
    raise ValueError(f"Unsupported frozen candidate: {name}")


def _infer(model, inputs: np.ndarray, kind: str, batch_size: int = 512) -> np.ndarray:
    model.eval()
    chunks = []
    with torch.inference_mode():
        for start in range(0, len(inputs), batch_size):
            batch = torch.from_numpy(np.asarray(inputs[start:start + batch_size], dtype=np.float32))
            output = model(batch)
            if kind == "autoencoder":
                scores = ((output - batch) ** 2).mean(dim=(1, 2))
            else:
                scores = torch.sigmoid(output)
            chunks.append(scores.detach().cpu().numpy().reshape(-1))
    return np.concatenate(chunks).astype(np.float64, copy=False)


def _roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Mann-Whitney ROC-AUC with average ranks for tied scores."""
    y = np.asarray(y_true, dtype=np.uint8)
    values = np.asarray(scores, dtype=np.float64)
    positives = int(y.sum())
    negatives = int(len(y) - positives)
    if positives == 0 or negatives == 0:
        raise ValueError("ROC-AUC requires both normal and anomaly samples")
    order = np.argsort(values, kind="mergesort")
    sorted_scores, sorted_y = values[order], y[order]
    positive_rank_sum = 0.0
    start = 0
    while start < len(y):
        end = start + 1
        while end < len(y) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        average_rank = ((start + 1) + end) / 2.0
        positive_rank_sum += average_rank * int(sorted_y[start:end].sum())
        start = end
    return float((positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives))


def _average_precision(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Step-wise average precision, grouping tied scores at each threshold."""
    y = np.asarray(y_true, dtype=np.uint8)
    values = np.asarray(scores, dtype=np.float64)
    positives = int(y.sum())
    if positives == 0:
        raise ValueError("PR-AUC requires at least one anomaly sample")
    order = np.argsort(-values, kind="mergesort")
    sorted_scores, sorted_y = values[order], y[order]
    tp = 0
    seen = 0
    previous_recall = 0.0
    average_precision = 0.0
    while seen < len(y):
        end = seen + 1
        while end < len(y) and sorted_scores[end] == sorted_scores[seen]:
            end += 1
        tp += int(sorted_y[seen:end].sum())
        precision = tp / end
        recall = tp / positives
        average_precision += (recall - previous_recall) * precision
        previous_recall = recall
        seen = end
    return float(average_precision)


def _evaluate_one(name: str, candidate: dict, metrics: dict, x: np.ndarray, y: np.ndarray,
                  n_features: int, window_size: int) -> dict:
    checkpoint = _load_checkpoint(C.ROOT / candidate["checkpoint"]["path"])
    frozen = candidate["selection"]
    threshold = float(frozen["threshold"])
    checkpoint_threshold = checkpoint.get("threshold")
    if checkpoint_threshold is not None and not np.isclose(float(checkpoint_threshold), threshold, rtol=0.0, atol=1e-12):
        raise ValueError(f"Saved threshold differs between checkpoint and freeze manifest for {name}")

    model, input_kind = _make_model(name, checkpoint, metrics, n_features, window_size)
    model.load_state_dict(checkpoint["state_dict"], strict=True)

    if input_kind == "row-wise":
        model_input, y_eval, row_offset = x, y, 0
    else:
        model_input, y_eval, ends = make_windows(x, y, window_size, "last")
        row_offset = int(ends[0]) if len(ends) else 0
    scores = _infer(model, model_input, input_kind)
    if len(scores) != len(y_eval):
        raise ValueError(f"Prediction/label length mismatch for {name}: {len(scores)} != {len(y_eval)}")

    y_pred = (scores >= threshold).astype(np.uint8)
    tn = int(np.sum((y_eval == 0) & (y_pred == 0)))
    fp = int(np.sum((y_eval == 0) & (y_pred == 1)))
    fn = int(np.sum((y_eval == 1) & (y_pred == 0)))
    tp = int(np.sum((y_eval == 1) & (y_pred == 1)))
    events = ReliabilityEngine(y_eval, y_pred).evaluate()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "model": name,
        "supervision": frozen.get("supervision", "—"),
        "input_representation": frozen.get("input_representation", "—"),
        "threshold": threshold,
        "threshold_rule": frozen.get("threshold_rule", "—"),
        "evaluation_split": "final_test",
        "evaluation_rows": int(len(y_eval)),
        "omitted_prefix_rows_for_window": row_offset,
        "anomalies": int(np.sum(y_eval)),
        "anomaly_prevalence": float(np.mean(y_eval)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(2 * precision * recall / (precision + recall)) if precision + recall else 0.0,
        "roc_auc": _roc_auc(y_eval, scores),
        "pr_auc": _average_precision(y_eval, scores),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "sample_false_positive_rate": float(fp / (fp + tn)) if fp + tn else None,
        "true_segments": int(events["true_segments"]),
        "detected_segments": int(events["detected_segments"]),
        "missed_segments": int(events["missed_segments"]),
        "detection_coverage": events["detection_coverage"],
        "mean_detection_delay_samples": events["mean_detection_delay"],
        "predicted_segments": int(events["predicted_segments"]),
        "false_alarm_segments": int(events["false_alarm_segments"]),
        "false_alarm_segment_rate": events["false_alarm_segment_rate"],
    }


def _write_outputs(results: list[dict], metadata: dict) -> dict:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "phase": "Final Test Evaluation",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_split": "final_test",
        "models_evaluated": [row["model"] for row in results],
        "model_selection": "All five frozen candidates were evaluated by explicit user request; no winner is selected by this report.",
        "threshold_policy": "Use each candidate's saved threshold unchanged from the freeze manifest; no threshold tuning on final_test.",
        "final_test_used": True,
        "final_test_file": "data/processed/machine-1-1/final_test.npz",
        "input_protocol": {
            "dataset": metadata["dataset"],
            "window_size": int(metadata["window_size"]),
            "label_mode": metadata["label_mode"],
            "window_boundaries": "Constructed within final_test only; no windows cross split boundaries.",
            "note": "MLP is row-wise. RNN/LSTM/CNN/Autoencoder use 30-step windows and omit the first 29 rows for complete-window alignment.",
        },
        "runtime": {"python_packages": {"torch": torch.__version__, "numpy": np.__version__}},
        "interpretation_caveats": [
            "This is a descriptive comparison across the five candidates requested; it does not designate a winner.",
            "Do not use this final-test comparison to tune thresholds, choose a model and then report the same test scores as an unbiased final estimate.",
            "MLP uses row-wise inputs (all final-test rows), while sequence and Autoencoder models use 30-step windows (the first 29 rows are omitted).",
            "The Autoencoder is normal-only trained and uses its saved label-free train_val threshold; supervised models use their saved sup_val-selected thresholds.",
            "Event-level reliability is computed on final-test predictions in memory using the existing ReliabilityEngine; per-sample predictions are not persisted.",
        ],
        "results": results,
    }
    json_path = RESULTS_DIR / "final_test_comparison.json"
    csv_path = RESULTS_DIR / "final_test_comparison.csv"
    md_path = RESULTS_DIR / "final_test_comparison.md"
    json_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")

    columns = list(results[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(results)

    table_fields = ("model", "evaluation_rows", "anomalies", "precision", "recall", "f1", "roc_auc", "pr_auc", "fp", "fn", "true_segments", "detected_segments", "missed_segments", "mean_detection_delay_samples", "predicted_segments", "false_alarm_segments", "false_alarm_segment_rate")
    lines = [
        "# Final Test Evaluation — All Five Frozen Candidates",
        "",
        "All five existing checkpoints were evaluated once on the held-out `final_test` split at the user's request. No model was trained, no predictions were saved, and no existing experiment result was overwritten.",
        "",
        "## Results",
        "",
        "| " + " | ".join(table_fields) + " |",
        "|" + "|".join(["---"] * len(table_fields)) + "|",
    ]
    for row in results:
        values = []
        for field in table_fields:
            value = row[field]
            values.append(f"{value:.4f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    lines += [
        "",
        "## Protocol",
        "",
        "Each model uses its saved checkpoint and threshold recorded in the Final Freeze manifest. Supervised thresholds were selected on `sup_val`; the Autoencoder threshold was selected label-free from `train_val` reconstruction errors. No threshold was adjusted using final-test labels.",
        "",
        "MLP is evaluated row-wise. RNN, LSTM, CNN, and Autoencoder are evaluated on 30-step windows labelled by their last row; the first 29 final-test rows therefore do not form a complete window for these models. Metric denominators are shown per model.",
        "",
        "The Reliability Engine event-level counts/delays are computed from these final-test predictions in memory. Per-sample predictions are not written to disk.",
        "",
        "## Interpretation limits",
        "",
        "This is a descriptive comparison across all five candidates, as requested; it does not name a winner. Because this final-test split has now been used for comparative evaluation, selecting a model based on these scores and presenting the same scores as an unbiased final estimate would introduce test-set selection bias.",
        "",
        "The candidates also differ in input representation and learning setup: MLP is row-wise, sequence models use temporal windows, and the Autoencoder is normal-only trained with label-free threshold selection. These scores should not be interpreted as isolating architecture alone.",
        "",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return {"json": str(json_path.relative_to(C.ROOT)), "csv": str(csv_path.relative_to(C.ROOT)), "markdown": str(md_path.relative_to(C.ROOT))}


def run_final_test_evaluation() -> dict:
    manifest = _read_json(MANIFEST_PATH)
    metadata = _read_json(META_PATH)
    candidates = manifest.get("candidates", [])
    expected_models = {"MLP", "RNN", "LSTM", "CNN", "Autoencoder"}
    if {item.get("model") for item in candidates} != expected_models:
        raise ValueError("Freeze manifest must contain exactly the five saved baseline candidates")
    _verify_saved_artifacts(candidates)

    final_split = metadata["splits"]["final_test"]
    with np.load(DATA_PATH, allow_pickle=False) as archive:
        if set(archive.files) != {"X", "y"}:
            raise ValueError(f"Unexpected final_test.npz fields: {archive.files}")
        x = np.asarray(archive["X"], dtype=np.float32)
        y = np.asarray(archive["y"], dtype=np.uint8)
    if x.ndim != 2 or x.shape != (int(final_split["rows"]), int(metadata["n_features"])):
        raise ValueError(f"Unexpected final-test X shape {x.shape}; expected rows x retained features from metadata")
    if y.ndim != 1 or len(y) != len(x) or not np.isin(y, (0, 1)).all():
        raise ValueError("Final-test labels must be a binary vector aligned with X")
    if int(y.sum()) != int(final_split["anomalies"]):
        raise ValueError("Final-test anomaly count does not match saved split metadata")

    by_name = {item["model"]: item for item in candidates}
    ordered_names = ["MLP", "RNN", "LSTM", "CNN", "Autoencoder"]
    results = [
        _evaluate_one(name, by_name[name], _read_json(C.ROOT / by_name[name]["metrics"]["path"]),
                      x, y, int(metadata["n_features"]), int(metadata["window_size"]))
        for name in ordered_names
    ]
    outputs = _write_outputs(results, metadata)
    return {"models_evaluated": ordered_names, "results": outputs}


if __name__ == "__main__":
    outcome = run_final_test_evaluation()
    print(json.dumps(outcome, indent=2))
