# Final Model Freeze — static readiness audit

## Audit outcome

The five architecture-comparison checkpoints are present and their SHA-256 fingerprints and paired metric/source references are recorded in `freeze_manifest.json`. This freezes an auditable **candidate set**, not a single selected final model.

**No single model is designated.** The existing `final_ann_comparison.md` explicitly says no definitive best model is declared, and the saved architecture comparison caveats state no winner is declared. This audit does not create a ranking from validation scores. A model must be explicitly selected before the project can claim a single Final Model Freeze.

## Checks

- **Checkpoint inventory:** MLP, RNN, LSTM, CNN, Autoencoder baseline checkpoint files exist. Hashes and byte sizes are in the manifest.
- **Metric-artifact consistency:** paired `metrics.json` and `architecture_comparison.json` values match for threshold, precision, recall, F1, ROC-AUC, PR-AUC and all confusion-matrix counts for all five models. No metrics were recomputed and no result files were altered.
- **Protocol traceability:** source config, preprocessing logic, metadata and training scripts record SMD machine-1-1, 30 retained features after dropping eight constants, source scaling unchanged, chronological splits, 30-step windows with `label_mode=last`, and split-boundary protection.
- **Selection provenance:** supervised checkpoint selection and thresholds used `sup_val`; Autoencoder checkpoint and threshold selection used label-free `train_val` reconstruction error after normal-only `train_fit` training.
- **Final-test boundary:** saved comparison reports `final_test_used=false`. `final_test.npz` was not opened, read, hashed, or evaluated.
- **Runtime status:** Python could not be launched by this Codex process, so `python run.py test` was not executed. Static inspection found that `run.py test` calls `window_test()`, which reads `dev.npz` and uses a synthetic zero array sized from metadata for its final-window shape assertion; it does not call `load_final_test()`.
- **Reliability limitation:** the Reliability Engine report says per-model saved validation prediction arrays are absent; model-level event reliability results are unavailable and were not fabricated.

## Freeze caveats

1. All supervised results are validation-selected and optimistic.
2. The project records one run per configuration (seed 42 where recorded), with only two contiguous validation anomaly segments and no repeated-seed uncertainty estimate.
3. Validation anomaly prevalence is about 45%, versus about 5.2% in the metadata-defined final-test split; the final-test data file was not accessed.
4. MLP row-wise input differs from sequence/window inputs, and the Autoencoder uses a distinct normal-only, label-free setup.

## Remaining decision

Before calling one checkpoint the final model, explicitly select the intended candidate and freeze the evaluation protocol, including carrying each saved threshold forward unchanged. No final-test evaluation is performed as part of this audit.

