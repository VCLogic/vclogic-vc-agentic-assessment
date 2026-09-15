# Rehearsal mode evaluation

Replays: 4; fallbacks: 0; questions: 5; accepted answers: 1; cost: $0.0983.

**Engineering-canary warning:** this sample is too small for significance claims or stable predictive-performance estimates.

| Method | Cases | Ins | Balanced accuracy | In precision | In recall | AP | ROC AUC | R@5 | R@10 | R@20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| canonical_v41_baseline | 4 | 2 | 0.750 | 1.000 | 0.500 | 0.833 | 0.750 | 1.000 | 1.000 | 1.000 |
| static_classifier | 4 | 2 | 0.500 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| v41_grounded_classifier | 4 | 2 | 0.500 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| v41_grounded_final | 4 | 2 | 0.750 | 1.000 | 0.500 | 0.833 | 0.750 | 1.000 | 1.000 | 1.000 |
