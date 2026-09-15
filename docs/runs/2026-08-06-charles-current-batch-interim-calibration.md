# Charles current-batch interim calibration

## Executive result

This interim diagnostic freezes 22 completed episodes from the active Charles Hudson LangGraph–Luna batch: 11 audited Ins and 11 audited Outs. It asks whether a supervised layer trained over frozen agent outputs can improve held-out pitch classification or ranking without additional LLM calls.

The strongest classification point estimate is the combined legacy rationale-core plus Phase 2 calibrator. It reaches **59.1% balanced accuracy**, **62.5% In precision**, **45.5% In recall**, and **52.6% In F1**. Its confusion counts are 5 true Ins, 3 false Ins, 8 true Outs, and 6 missed Ins. This is more selective than the raw any-check endpoint, but its 40.9%–77.3% bootstrap interval for balanced accuracy includes chance.

For ranking, Phase 2 calibration is the strongest learned method at **0.625 average precision** and **0.612 ROC AUC**. It places 3 Ins in the top 5, 6 in the top 10, and all 11 in the top 20. Raw standard-check likelihood has a higher AP point estimate, **0.662**, while Phase 2 calibration has the higher ROC AUC. Neither difference is stable enough to claim superiority on 22 prevalence-enriched cases.

## Evaluation protocol

Each learned score is out of sample. Five repetitions of stratified five-fold outer evaluation are used, with training-only regularization selection, Platt score calibration, and threshold selection. Every validation fold contains both labels. This avoids the outer-fold class-count artifact previously identified in pooled leave-one-out evaluation.

The frozen snapshot and source hashes are stored with the result. The active inference batch was neither stopped nor modified. The first five frozen investigations predate the addition of `taxonomy_dispositions` to the v3 schema; the calibration loader verifies their original hashes and reconstructs those dispositions in memory from their already-frozen activated and rejected candidate lists. No source artifact is rewritten.

## Classification results

| Method | Balanced accuracy | In precision | In recall | In F1 | TP / FP / TN / FN |
|---|---:|---:|---:|---:|---:|
| **Rationale core + Phase 2** | **59.1%** | **62.5%** | 45.5% | **52.6%** | 5 / 3 / 8 / 6 |
| Phase 2 | 54.5% | 55.6% | 45.5% | 50.0% | 5 / 4 / 7 / 6 |
| Raw any-check | 54.5% | 52.6% | **90.9%** | **66.7%** | 10 / 9 / 2 / 1 |
| Raw ranking score as decision | 50.0% | 50.0% | 63.6% | 56.0% | 7 / 7 / 4 / 4 |
| Always Out | 50.0% | 0% | 0% | 0% | 0 / 0 / 11 / 11 |
| Raw standard-check | 50.0% | 0% | 0% | 0% | 0 / 0 / 11 / 11 |
| Rationale core | 45.5% | 45.5% | 45.5% | 45.5% | 5 / 6 / 5 / 6 |

The combined calibrator's true Ins are episodes 41, 102, 104, 133, and 135. Its false Ins are 20, 37, and 38. It misses episodes 68, 77, 78, 83, 127, and 143.

The raw any-check remains the high-recall screen: it finds 10 of 11 Ins but sends 19 of 22 pitches to review. The calibrated combined model sends only 8 pitches to review and finds 5 Ins. These are different operational trade-offs rather than one endpoint unambiguously dominating the other.

## Ranking results

The prevalence/AP reference is 0.500 because the round-robin snapshot is exactly balanced.

| Ranking signal | Average precision | ROC AUC | Top 5 Ins | Top 10 Ins | Top 20 Ins |
|---|---:|---:|---:|---:|---:|
| Raw standard-check likelihood | **0.662** | 0.595 | 3 | 6 | 11 |
| **Phase 2 calibration** | **0.625** | **0.612** | 3 | 6 | 11 |
| Raw ranking score | 0.577 | 0.566 | 2 | 6 | 10 |
| Rationale core | 0.532 | 0.405 | 3 | 5 | 9 |
| Raw any-check likelihood | 0.502 | 0.496 | 2 | 5 | 10 |
| Rationale core + Phase 2 | 0.454 | 0.413 | 1 | 5 | 10 |

The Phase 2 calibrator's top five are episodes 135 (In), 54 (Out), 37 (Out), 77 (In), and 41 (In). Raw standard-check likelihood ranks episodes 127, 41, and 78—all Ins—as its top three. The combined calibrator improves the classification thresholding trade-off but does not improve ranking.

## Interpretation

The post-prediction layer is technically viable and appears capable of correcting part of the agents' threshold problem. The combined model reduces raw any-check false positives from nine to three, at the cost of reducing recovered Ins from ten to five. Phase 2 contains useful ranking information even though its direct standard-check decision is always Out.

The evidence is not yet strong enough to select a final model. The snapshot is small, artificially balanced by round-robin execution, and contains a documented retrieval-configuration transition after its first five records. Bootstrap intervals are wide: the combined classifier's balanced-accuracy interval includes 50%, and the Phase 2 AP interval spans approximately 0.468–0.862. Precision and AP will also change when the final population returns to its naturally lower In prevalence.

The appropriate conclusion is therefore: **continue collecting the frozen agent outputs, retain both the raw high-recall screen and the calibrated selective endpoint, and rerun the identical locked protocol on the completed 86-episode population.**

## Reproducibility

- Metrics: `outputs/charles-current-batch-interim-calibration-2026-08-06/metrics.json`
- Held-out predictions: `outputs/charles-current-batch-interim-calibration-2026-08-06/predictions.csv`
- Rankings: `outputs/charles-current-batch-interim-calibration-2026-08-06/rankings.csv`
- Method manifest: `outputs/charles-current-batch-interim-calibration-2026-08-06/method-manifest.json`
- Snapshot provenance: `outputs/charles-current-batch-interim-calibration-2026-08-06/snapshot/snapshot-provenance.json`
- Command: `uv run python scripts/analyze_current_batch_calibration.py --status-path outputs/openrouter-luna-charles-two-iterations-semantic-full-2026-08-05/batch-status.json --labels-path evaluation/labels/charles_pitch_window_decisions.json --output-root outputs/charles-current-batch-interim-calibration-2026-08-06 --repeats 5 --splits 5 --bootstrap-iterations 2000`
