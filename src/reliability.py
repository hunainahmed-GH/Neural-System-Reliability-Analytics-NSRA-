"""Reliability Engine: event-level (segment-level) analysis of anomaly predictions.

Two layers live in this module:

* Legacy helpers (`segments`, `reliability_stats`) -- unchanged. They are used by the
  training scripts and EDA and produced the `reliability` blocks already stored in the
  experiment `metrics.json` files. Their `false_alarm_rate` is a SAMPLE-LEVEL false
  positive rate (FP / normal samples).
* The Reliability Engine (`find_contiguous_segments`, `ReliabilityEngine`) -- event-level
  analysis of `y_true` / `y_pred`.

IMPORTANT -- `false_alarm_segment_rate`
    false_alarm_segment_rate = false_alarm_segments / predicted_segments

    This is an EVENT-LEVEL SEGMENT PROPORTION: the share of *predicted anomaly segments*
    that do not overlap any true anomaly segment. It is NOT the sample-level false
    positive rate (FP / (FP + TN)) and the two must not be compared or substituted.

Conventions
    * Segments are (start, end) with BOTH ends inclusive, 0-based sample indices.
    * A predicted segment detects a true segment iff they share at least one sample.
    * Detection delay = first sample of the true segment that is predicted anomalous
      minus the true segment start (>= 0). A predicted segment that begins before the
      true segment therefore gives delay 0.
    * Ratios with a zero denominator are undefined and reported as None
      (never silently 0.0 or 1.0):
        - detection_coverage is None when there are no true segments,
        - false_alarm_segment_rate is None when there are no predicted segments.
"""
import numpy as np


# --------------------------------------------------------------------------- legacy
def segments(y):
    """(start, end_inclusive) of contiguous anomaly runs."""
    y = np.asarray(y).astype(int)
    d = np.diff(np.r_[0, y, 0])
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0] - 1))


def reliability_stats(y, yhat):
    """Segment-level reliability view: detected segments, detection delay, false-alarm rate.

    NOTE: `false_alarm_rate` here is the sample-level FPR over normal samples.
    """
    segs = segments(y)
    yhat = np.asarray(yhat).astype(int)
    delays = []
    for a, b in segs:
        hit = np.where(yhat[a:b + 1] == 1)[0]
        if len(hit):
            delays.append(int(hit[0]))
    normal = np.asarray(y) == 0
    return {"n_segments": len(segs), "detected_segments": len(delays),
            "segment_recall": len(delays) / max(len(segs), 1),
            "mean_detection_delay": float(np.mean(delays)) if delays else None,
            "false_alarm_rate": float(yhat[normal].mean()) if normal.any() else 0.0,
            "mean_segment_len": float(np.mean([b - a + 1 for a, b in segs])) if segs else 0.0}


# ------------------------------------------------------------------- Reliability Engine
def find_contiguous_segments(binary_sequence):
    """Return contiguous runs of 1s as a list of (start, end) tuples, end inclusive.

    Empty input (or input with no 1s) returns []. Indices are plain Python ints.
    """
    arr = np.asarray(binary_sequence)
    if arr.size == 0:
        return []
    if arr.ndim != 1:
        raise ValueError(f"binary_sequence must be 1-D, got shape {arr.shape}")
    arr = (arr != 0).astype(np.int8)
    d = np.diff(np.concatenate(([0], arr, [0])))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1) - 1
    return [(int(s), int(e)) for s, e in zip(starts, ends)]


def _overlaps(a, b):
    """True if inclusive segments a and b share at least one sample."""
    return a[0] <= b[1] and b[0] <= a[1]


class ReliabilityEngine:
    """Event-level reliability analysis of binary predictions against binary labels.

    Parameters
    ----------
    y_true, y_pred : 1-D array-likes of 0/1 with equal length (ValueError otherwise).

    `evaluate()` returns a dict with:
        true_segments, detected_segments, missed_segments   (counts)
        detection_coverage       detected / true segments (None if no true segments)
        detection_delays         list, one delay per detected true segment
        mean_detection_delay     mean of delays (None if nothing detected)
        predicted_segments       number of predicted anomaly segments
        false_alarm_segments     predicted segments overlapping NO true segment
        false_alarm_segment_rate false_alarm_segments / predicted_segments
                                 (EVENT-LEVEL segment proportion, NOT sample-level FPR;
                                  None if there are no predicted segments)
        plus the per-segment lists (true/predicted/detected/missed/false-alarm segments).
    """

    def __init__(self, y_true, y_pred):
        yt, yp = np.asarray(y_true), np.asarray(y_pred)
        if yt.ndim != 1 or yp.ndim != 1:
            raise ValueError(f"y_true and y_pred must be 1-D, got {yt.shape} and {yp.shape}")
        if len(yt) != len(yp):
            raise ValueError(f"length mismatch: len(y_true)={len(yt)} != len(y_pred)={len(yp)}")
        for name, a in (("y_true", yt), ("y_pred", yp)):
            if a.size and not np.isin(a, (0, 1, False, True)).all():
                raise ValueError(f"{name} must be binary (0/1)")
        self.y_true = yt.astype(np.int8)
        self.y_pred = yp.astype(np.int8)

    def evaluate(self):
        true_segs = find_contiguous_segments(self.y_true)
        pred_segs = find_contiguous_segments(self.y_pred)

        detected, missed, delays = [], [], []
        for t in true_segs:
            overlap = np.flatnonzero(self.y_pred[t[0]:t[1] + 1])
            if overlap.size:
                detected.append(t)
                delays.append(int(overlap[0]))  # first overlapping predicted sample - start
            else:
                missed.append(t)

        false_alarms = [p for p in pred_segs if not any(_overlaps(p, t) for t in true_segs)]

        n_true, n_pred = len(true_segs), len(pred_segs)
        return {
            "true_segments": n_true,
            "detected_segments": len(detected),
            "missed_segments": len(missed),
            "detection_coverage": len(detected) / n_true if n_true else None,
            "detection_delays": delays,
            "mean_detection_delay": float(np.mean(delays)) if delays else None,
            "predicted_segments": n_pred,
            "false_alarm_segments": len(false_alarms),
            "false_alarm_segment_rate": len(false_alarms) / n_pred if n_pred else None,
            "true_segment_list": true_segs,
            "predicted_segment_list": pred_segs,
            "detected_segment_list": detected,
            "missed_segment_list": missed,
            "false_alarm_segment_list": false_alarms,
        }
