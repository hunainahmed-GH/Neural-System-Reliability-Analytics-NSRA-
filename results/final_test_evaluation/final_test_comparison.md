# Final Test Evaluation — All Five Frozen Candidates

All five existing checkpoints were evaluated once on the held-out `final_test` split at the user's request. No model was trained, no predictions were saved, and no existing experiment result was overwritten.

## Results

| model | evaluation_rows | anomalies | precision | recall | f1 | roc_auc | pr_auc | fp | fn | true_segments | detected_segments | missed_segments | mean_detection_delay_samples | predicted_segments | false_alarm_segments | false_alarm_segment_rate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MLP | 8042 | 416 | 0.1615 | 0.7043 | 0.2628 | 0.8707 | 0.4009 | 1521 | 123 | 4 | 4 | 0 | 0.0000 | 246 | 235 | 0.9553 |
| RNN | 8013 | 416 | 0.1502 | 0.6995 | 0.2472 | 0.8600 | 0.4682 | 1647 | 125 | 4 | 4 | 0 | 0.2500 | 22 | 17 | 0.7727 |
| LSTM | 8013 | 416 | 0.1737 | 0.7764 | 0.2838 | 0.9197 | 0.4924 | 1537 | 93 | 4 | 4 | 0 | 0.2500 | 41 | 36 | 0.8780 |
| CNN | 8013 | 416 | 0.1407 | 0.8125 | 0.2398 | 0.8814 | 0.3923 | 2065 | 78 | 4 | 4 | 0 | 0.2500 | 13 | 8 | 0.6154 |
| Autoencoder | 8013 | 416 | 0.2068 | 0.6418 | 0.3128 | 0.9015 | 0.2821 | 1024 | 149 | 4 | 4 | 0 | 0.0000 | 33 | 23 | 0.6970 |

## Protocol

Each model uses its saved checkpoint and threshold recorded in the Final Freeze manifest. Supervised thresholds were selected on `sup_val`; the Autoencoder threshold was selected label-free from `train_val` reconstruction errors. No threshold was adjusted using final-test labels.

MLP is evaluated row-wise. RNN, LSTM, CNN, and Autoencoder are evaluated on 30-step windows labelled by their last row; the first 29 final-test rows therefore do not form a complete window for these models. Metric denominators are shown per model.

The Reliability Engine event-level counts/delays are computed from these final-test predictions in memory. Per-sample predictions are not written to disk.

## Interpretation limits

This is a descriptive comparison across all five candidates, as requested; it does not name a winner. Because this final-test split has now been used for comparative evaluation, selecting a model based on these scores and presenting the same scores as an unbiased final estimate would introduce test-set selection bias.

The candidates also differ in input representation and learning setup: MLP is row-wise, sequence models use temporal windows, and the Autoencoder is normal-only trained with label-free threshold selection. These scores should not be interpreted as isolating architecture alone.
