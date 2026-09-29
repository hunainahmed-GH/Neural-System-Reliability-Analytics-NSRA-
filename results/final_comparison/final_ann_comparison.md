# NSRA — Final ANN Comparison (consolidation of existing results)

Dataset: SMD machine-1-1 (30 features after dropping 8 constant ones). Source of every number: existing `metrics.json` / `comparison.csv` files. Nothing was retrained, no inference was run, `final_test.npz` was not loaded, and no existing result file was modified.

**No definitive best model is declared here.** All figures are on `sup_val` (see caveats). The held-out `final_test` has not been used.

## 1. Evaluation protocol per model

| Model | Input | Architecture | Params | Supervision | Threshold rule |
|---|---|---|---:|---|---|
| MLP | row-wise, 30 features | 30 → Dense(64) → ReLU → Dense(1) | 2,049 | supervised, BCE + pos_weight | max-F1 on `sup_val` (labels) |
| RNN | 30×30 window | Elman RNN(30→64, tanh) → last step → Dense(1) | 6,209 | supervised, BCE + pos_weight | max-F1 on `sup_val` (labels) |
| LSTM | 30×30 window | LSTM(30→64) → last step → Dense(1) | 24,641 | supervised, BCE + pos_weight | max-F1 on `sup_val` (labels) |
| CNN | 30×30 window | Conv1d(30→64,k3)–ReLU–Conv1d(64→64,k3)–ReLU–GAP–Dense(1) | 18,241 | supervised, BCE + pos_weight | max-F1 on `sup_val` (labels) |
| Autoencoder | 30×30 window, flattened (900) | FC 900-128-32-128-900, MSE | 239,780 | unsupervised, normal-only `train_fit` | 99th pct of `train_val` recon. error (no labels) |

Parameter counts are computed analytically from the model definitions in `src/models/`. Supervised models used checkpoint selection by `sup_val` PR-AUC (Adam, lr 1e-3, batch 256, patience 10, max 100 epochs). The Autoencoder's checkpoint was selected by `train_val` reconstruction MSE, without labels.

## 2. Architecture results on `sup_val`

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC | FP | FN | FP+FN | False-alarm rate | MCC | Segments detected | Mean delay |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| MLP | 0.9814 | 0.9839 | 0.9826 | 0.9968 | 0.9927 | 22 | 19 | 41 | 0.0150 | 0.9686 | 2/2 | 0 |
| RNN | 0.9958 | 1.0000 | 0.9979 | 1.0000 | 1.0000 | 5 | 0 | 5 | 0.0035 | 0.9961 | 2/2 | 0 |
| LSTM | 0.9907 | 0.9898 | 0.9902 | 0.9995 | 0.9993 | 11 | 12 | 23 | 0.0077 | 0.9822 | 2/2 | 0 |
| CNN | 0.9865 | 0.9941 | 0.9903 | 0.9983 | 0.9976 | 16 | 7 | 23 | 0.0111 | 0.9823 | 2/2 | 0 |
| Autoencoder | 0.8879 | 0.7063 | 0.7868 | 0.9126 | 0.8581 | 105 | 346 | 451 | 0.0731 | 0.6569 | n/a* | n/a* |

\* The Autoencoder `metrics.json` has no reliability block (segment recall / detection delay). Shown as n/a, not estimated.

Evaluation sets: MLP n = 2,643 rows (1,465 normal / 1,178 anomalous); windowed models n = 2,614 windows (1,436 normal / 1,178 anomalous). The first 29 rows of `sup_val` have no full window, so windowed models see 29 fewer normal samples. FP+FN, false-alarm rate, and MCC are derived from the stored confusion matrices; stored precision/recall/F1 and false-alarm rates were re-checked against them and match.

Plots: `plots/architecture_metrics.png`, `plots/architecture_error_counts.png`, `plots/architecture_confusion_matrices.png`.

## 3. What the numbers do and do not say

- **Supervised group (MLP, RNN, LSTM, CNN):** all four reach F1 ≥ 0.98 on `sup_val` and detect both anomaly segments with zero delay. Spread between them is 5–41 total errors out of ~2.6k samples. Thresholds were tuned on the same split, so these gaps are optimistic and are not evidence of a reliable ranking.
- **MLP vs windowed models:** the comparison mixes two factors — architecture and input representation (single row vs 30-step context). The results cannot separate them.
- **RNN vs LSTM vs CNN:** RNN has the fewest errors here (5, versus 23 each for LSTM and CNN). That is a single run per model, with thresholds tuned on the same split and only two anomaly segments, so it is not enough to rank them.
- **Autoencoder:** clearly lower (F1 0.787, ROC-AUC 0.913), but it is a different task setting: no labels were used for training, checkpointing, or thresholding. Its recall is 0.706 at the label-free 99th-percentile threshold; ROC-AUC (threshold-free, 0.913) is the fairer number, and it also trails the supervised models on this split. Its mean `sup_val` reconstruction error (0.00201) is about 10× the `train_val` mean (0.00020), so the score does separate anomalies; the operating point is the weak part. This is an observation, not a claim that a different threshold would fix it.
- **Not directly comparable:** the Autoencoder versus the supervised group, and any cross-model precision/PR-AUC comparison, given the prevalence difference below.

## 4. MLP experiment comparison (all on `sup_val`, one run per config)

Each group varies one factor from the shared reference (ReLU, Adam, no regularization, lr 1e-3, hidden 64; marked *). The reference row repeats in every group. 21 rows, 17 unique configurations.

| Group | Config | F1 | PR-AUC | ROC-AUC | FP | FN | Best / run epochs |
|---|---|---:|---:|---:|---:|---:|---|
| Activation | relu* | 0.9826 | 0.9927 | 0.9968 | 22 | 19 | 94 / 100 |
| | leakyrelu | 0.9822 | 0.9933 | 0.9968 | 20 | 22 | 99 / 100 |
| | tanh | 0.9774 | 0.9905 | 0.9957 | 21 | 32 | 40 / 50 |
| Optimizer | sgd | 0.8538 | 0.9260 | 0.9311 | 135 | 200 | 14 / 24 |
| | sgd_momentum | 0.9061 | 0.9730 | 0.9715 | 98 | 121 | 100 / 100 |
| | rmsprop | 0.9830 | 0.9956 | 0.9972 | 20 | 20 | 99 / 100 |
| | adam* | 0.9826 | 0.9927 | 0.9968 | 22 | 19 | 94 / 100 |
| Regularization | none* | 0.9826 | 0.9927 | 0.9968 | 22 | 19 | 94 / 100 |
| | l2 (1e-4) | 0.9839 | 0.9930 | 0.9970 | 22 | 16 | 94 / 100 |
| | dropout (0.2) | 0.9826 | 0.9961 | 0.9973 | 21 | 20 | 98 / 100 |
| | l2 + dropout | 0.9842 | 0.9966 | 0.9976 | 12 | 25 | 100 / 100 |
| Learning rate | 1e-4 | 0.8331 | 0.9021 | 0.9083 | 210 | 187 | 2 / 12 |
| | 3e-4 | 0.9596 | 0.9879 | 0.9894 | 7 | 85 | 99 / 100 |
| | 1e-3* | 0.9826 | 0.9927 | 0.9968 | 22 | 19 | 94 / 100 |
| | 3e-3 | 0.9864 | 0.9984 | 0.9987 | 16 | 16 | 74 / 84 |
| | 1e-2 | 0.9877 | 0.9986 | 0.9989 | 16 | 13 | 31 / 41 |
| Hidden size | 16 (513 params) | 0.9694 | 0.9898 | 0.9946 | 19 | 52 | 92 / 100 |
| | 32 (1,025) | 0.9813 | 0.9923 | 0.9964 | 22 | 22 | 99 / 100 |
| | 64* (2,049) | 0.9826 | 0.9927 | 0.9968 | 22 | 19 | 94 / 100 |
| | 128 (4,097) | 0.9835 | 0.9949 | 0.9974 | 24 | 15 | 99 / 100 |
| | 256 (8,193) | 0.9842 | 0.9974 | 0.9981 | 15 | 22 | 100 / 100 |

Plot: `plots/mlp_experiments_f1_prauc.png` (y-axis truncated at 0.8).

Observations, kept descriptive:
- **Large effects:** plain SGD, SGD with momentum, and lr 1e-4 are clearly worse (F1 0.83–0.91). lr 3e-4 and hidden 16 are also visibly lower (F1 0.96–0.97).
- **Small effects:** among activation (ReLU/LeakyReLU), RMSprop/Adam, and the four regularization settings, F1 spans 0.9822–0.9842 — a difference of a few FP/FN samples on one seed, well within plausible noise.
- **Learning rate:** lr 3e-3 and 1e-2 score slightly higher than the 1e-3 reference (PR-AUC 0.998–0.999 vs 0.993), and both early-stopped before the epoch cap. This is a single-run, `sup_val`-tuned observation and should not be treated as an established improvement.
- **Under-trained runs:** 12 of the 17 unique configurations ran to the 100-epoch cap with the best epoch at or near the end (including sgd_momentum at 100/100). For those, the ranking may reflect training budget as much as the factor being varied.

## 5. Caveats that apply to everything above

1. Supervised thresholds (max-F1) and checkpoint selection used `sup_val`, so `sup_val` classification metrics are optimistic. The same applies to every MLP experiment.
2. `sup_val` has only **2 contiguous anomaly segments** (mean length 589 rows). Row/window metrics are heavily autocorrelated, so the effective sample size is much smaller than n ≈ 2.6k, and near-perfect scores (RNN ROC-AUC/PR-AUC ≈ 1.000) should be read with caution.
3. **Prevalence shift:** `sup_val` is ~45% anomalous; `final_test` is 416/8,042 = 5.2% (from the split definition, not from loading the file). Precision, F1, and PR-AUC depend on prevalence and are expected to drop on `final_test` even for an unchanged model; ROC-AUC, recall, and false-alarm rate transfer more directly. The documented train-vs-dev drift may also matter.
4. Windowed and row-wise evaluation sets differ by 29 normal samples.
5. One run per configuration; no seed repetitions or confidence intervals.
6. The Autoencoder result reflects a label-free operating point; the supervised results reflect label-tuned operating points.

## 6. Suggested handling before `final_test` (for discussion, not done)

- Freeze the full protocol first: which models, which thresholds (carry the `sup_val` thresholds over unchanged; do not re-tune on `final_test`), and which metrics to report.
- Report threshold-free (ROC-AUC, PR-AUC) and threshold-dependent (recall, false-alarm rate, segment recall/delay) metrics side by side, since prevalence differs.
- Evaluate all five models once, and treat any ranking as provisional given a single seed and two validation segments.

## 7. Files produced

`architecture_comparison.{json,csv}`, `mlp_experiment_comparison.{json,csv}`, `final_ann_comparison.md`, `plots/*.png` (4 figures) — all under `experiments/results/final_comparison/`. Generator: `src/final_comparison.py`, callable via `python run.py final_comparison` (new command added to `run.py`).
