# Charles Hudson expanded calibration benchmark

## Executive conclusion

All proposed complex models were implemented and evaluated locally over the 86 frozen Charles Hudson episodes. **None provides a defensible improvement over the simpler baselines after correcting the outer evaluation protocol.**

Under the primary strict labels (17 Ins, 69 Outs):

- The strongest classifier is the **legacy Phase 1 rationale core**, at **56.2% balanced accuracy**. It identifies 10 of 17 Ins but generates 32 false positives.
- The strongest learned ranking method is the **Phase 2 calibration**, at **0.282 average precision**, compared with 0.198 In prevalence. It finds 1 In in the top 5, 1 in the top 10, and 5 in the top 20.
- The strongest newly tested classifier is the RBF SVM at only 53.1% balanced accuracy.
- The strongest newly tested ranker is the two-stage diamond model at 0.255 AP, below calibrated Phase 2.

The uncertainty intervals are wide. The legacy Phase 1 classifier's 95% bootstrap interval for balanced accuracy is 43.7%–69.4%; the Phase 2 ranking AP interval is 0.182–0.463. These results do not establish generalization beyond this repeatedly examined development population.

No LLM or API calls were used. Every method consumes the same frozen, hash-verified Phase 1 and Phase 2 artifacts.

## Critical correction to the evaluation

The first expanded diagnostic used nested leave-one-out evaluation, matching the earlier semantic-calibration experiment. It produced an apparently exceptional RBF-SVM ranking result: 0.643 AP and nine Ins in the top ten. That result was rejected before the sensitivity analysis completed.

The reason was an outer-fold class-count artifact. With 17 total Ins, holding out an In left 16 positive training cases; holding out an Out left 17. Hyperparameter selection responded sharply to that difference:

- `C=0.1, gamma=scale` was selected for 7 held Ins and 0 held Outs;
- `C=10, gamma=scale` was selected for 0 held Ins and 41 held Outs.

Thus the training population itself disclosed the held episode's class to fold-specific model selection. Raw SVM decision margins from separately tuned folds also lacked a common ranking scale. The apparent RBF improvement was therefore an evaluation artifact rather than credible pitch discrimination.

The final benchmark corrects both issues:

1. It uses **five repeats of stratified five-fold outer evaluation**. Every validation fold contains both classes, and each episode receives five genuinely out-of-sample predictions.
2. The strict fold assignments are reused unchanged for the conditional-interest sensitivity labels.
3. Model scores are mapped through **training-only Platt calibration** before predictions from different outer models are pooled.
4. Hyperparameters, calibration, and decision thresholds are selected using only each outer training population.
5. Every learned prediction records all five training folds, permitting direct verification that the target episode was absent from each.

This corrected benchmark also reruns the five simpler learned baselines. It does not reuse their earlier pooled leave-one-out predictions. The 2026-08-02 semantic-calibration report is now explicitly marked superseded.

## Models compared

### Simpler learned baselines

- Legacy Phase 1 rationale core: aggregate direction, salience, confidence, conflict, and unanswered-question features.
- Semantic Phase 1: taxonomy labels, evidence status, constraint semantics, severity, and deal-context indicators.
- Phase 2 calibration: structured decisions, likelihoods, risks, market gate, rationale assessments, quality findings, and check-tier fields.
- Legacy Phase 1 plus Phase 2.
- Semantic Phase 1 plus Phase 2.

These use standardized, class-balanced logistic regression with training-only regularization selection.

### Newly tested complex models

- **Cross-fitted late fusion:** separate Semantic Phase 1 and Phase 2 base scores, their product and difference, raw any-check likelihood, and a logistic meta-learner.
- **Elastic-net combined model:** sparse logistic regression over 274 semantic Phase 1 plus Phase 2 features.
- **Shallow gradient boosting:** constrained nonlinear trees over the combined representation.
- **Linear SVM** and **RBF SVM:** class-balanced support-vector models over combined features.
- **Pairwise ranker:** learns positive-minus-negative feature differences to prioritize Ins rather than optimize probabilities.
- **Two-stage diamond model:** a training-selected mixture of a high-recall Phase 1 candidate score and a Phase 2 synthesis score.

### Fixed references

- Always Out.
- Raw any-check decision and likelihood.
- Raw standard-check decision and likelihood.
- Raw Phase 2 ranking score.

## Strict-label results

Population: **86 episodes: 17 Ins and 69 Outs**. AP prevalence baseline: **0.198**.

| Method | Balanced accuracy | In precision | In recall | In F1 | AP | Top 5 / 10 / 20 Ins |
|---|---:|---:|---:|---:|---:|---:|
| Late fusion | 48.0% | 16.7% | 17.6% | 17.1% | 0.177 | 0 / 0 / 4 |
| Elastic-net combined | 39.4% | 13.3% | 35.3% | 19.4% | 0.154 | 0 / 0 / 2 |
| Shallow gradient boosting | 43.7% | 14.7% | 29.4% | 19.6% | 0.172 | 0 / 0 / 3 |
| Linear SVM | 48.3% | 18.9% | 58.8% | 28.6% | 0.156 | 0 / 1 / 2 |
| RBF SVM | 53.1% | 25.0% | 23.5% | 24.2% | 0.178 | 0 / 1 / 1 |
| Pairwise ranker | 51.1% | 20.5% | 47.1% | 28.6% | 0.195 | 0 / 1 / 4 |
| Two-stage diamond | 47.3% | 16.7% | 23.5% | 19.5% | 0.255 | 1 / 2 / 4 |
| **Legacy Phase 1** | **56.2%** | 23.8% | **58.8%** | **33.9%** | 0.220 | 0 / 2 / 5 |
| Semantic Phase 1 | 43.1% | 15.6% | 41.2% | 22.6% | 0.163 | 1 / 1 / 2 |
| **Phase 2 calibration** | 55.3% | **26.1%** | 35.3% | 30.0% | **0.282** | 1 / 1 / 5 |
| Legacy Phase 1 + Phase 2 | 50.3% | 20.0% | 35.3% | 25.5% | 0.257 | 1 / 3 / **6** |
| Semantic Phase 1 + Phase 2 | 40.2% | 14.3% | 41.2% | 21.2% | 0.169 | 0 / 1 / 2 |
| Raw any-check | 50.6% | 20.0% | **82.4%** | 32.2% | 0.269 | 1 / **3** / **7** |
| Raw standard-check | 50.0% | 0.0% | 0.0% | 0.0% | 0.231 | 1 / 2 / 4 |
| Raw ranking score | 47.8% | 19.0% | 94.1% | 31.7% | 0.186 | 0 / 0 / 0 |

### Classification interpretation

Legacy Phase 1 is the point-estimate winner, but it remains a high-recall screen rather than a selective final classifier:

- 10 true Ins;
- 32 false Ins;
- 37 true Outs;
- 7 missed Ins.

Its missed Ins are:

- `68-dogs-dating-a-match-made-in-heaven`
- `78-got-goals-grab-a-cru`
- `102-tether-bodyguard-of-the-grid`
- `104-dressd-red-carpet-or-red-ocean`
- `135-thoras-ai-the-twin-effect`
- `148-esai`
- `149-dopl`

Its 23.8% In precision is only modestly above the 19.8% population prevalence. The bootstrap interval for balanced accuracy includes chance, so 56.2% should not be portrayed as a stable predictive achievement.

### Ranking interpretation

Calibrated Phase 2 has the best learned AP, 0.282. The five actual Ins in its top 20 are:

- rank 1: `102-tether-bodyguard-of-the-grid`;
- rank 11: `68-dogs-dating-a-match-made-in-heaven`;
- rank 13: `143-vital-audio`;
- rank 14: `77-sell-online-or-go-door-to-door`;
- rank 20: `170-original-sunshine-the-best-bagel-youve-never-heard-of`.

Raw any-check has slightly lower AP, 0.269, but recovers more Ins at the larger review budgets: 3 in the top 10 and 7 in the top 20. Phase 2 calibration therefore improves the global precision-recall ordering slightly while not improving the operational top-10/top-20 queue. The difference is small and its uncertainty overlaps substantially.

Legacy Phase 1 plus Phase 2 reaches six Ins in the top 20, while the two-stage model reaches four. This is evidence that Phase 2 contains some ranking signal, but not evidence that the newly tested fusion mechanisms exploit it better.

## Conditional-interest sensitivity results

Episodes 41 and 127 are relabeled In, giving **19 Ins and 67 Outs**. AP prevalence becomes **0.221**.

| Method | Balanced accuracy | In precision | In recall | In F1 | AP | Top 5 / 10 / 20 Ins |
|---|---:|---:|---:|---:|---:|---:|
| Late fusion | 44.9% | 17.6% | 31.6% | 22.6% | 0.168 | 1 / 2 / 2 |
| Elastic-net combined | 46.7% | 18.5% | 26.3% | 21.7% | 0.220 | 0 / 2 / 5 |
| Shallow gradient boosting | 40.8% | 14.3% | 26.3% | 18.5% | 0.174 | 0 / 0 / 2 |
| Linear SVM | 47.2% | 20.4% | 52.6% | 29.4% | 0.213 | 1 / 2 / 3 |
| RBF SVM | 44.9% | 17.6% | 31.6% | 22.6% | 0.215 | 2 / 3 / 4 |
| Pairwise ranker | 50.9% | 22.9% | 42.1% | 29.6% | 0.225 | 1 / 2 / 5 |
| Two-stage diamond | 32.6% | 10.9% | 26.3% | 15.4% | 0.192 | 1 / 1 / 2 |
| **Legacy Phase 1** | **58.1%** | **27.1%** | 68.4% | **38.8%** | 0.251 | 1 / 1 / 6 |
| Semantic Phase 1 | 53.6% | 24.1% | 68.4% | 35.6% | 0.213 | 1 / 1 / 3 |
| Phase 2 calibration | 46.0% | 17.9% | 26.3% | 21.3% | 0.204 | 1 / 1 / 3 |
| Legacy Phase 1 + Phase 2 | 47.9% | 20.0% | 31.6% | 24.5% | 0.205 | 0 / 1 / 5 |
| Semantic Phase 1 + Phase 2 | 49.5% | 21.8% | 63.2% | 32.4% | 0.211 | 1 / 2 / 4 |
| **Raw any-check** | 51.8% | 22.9% | **84.2%** | 36.0% | **0.331** | **2 / 4 / 8** |
| Raw standard-check | 50.0% | 0.0% | 0.0% | 0.0% | 0.269 | 1 / 3 / 5 |
| Raw ranking score | 48.1% | 21.4% | 94.7% | 35.0% | 0.196 | 0 / 0 / 0 |

The conditional-interest definition strengthens the legacy Phase 1 classifier point estimate to 58.1% balanced accuracy, with 13 true Ins, 35 false Ins, 32 true Outs, and 6 missed Ins. Its 95% balanced-accuracy interval is 46.1%–70.1%.

Raw any-check becomes the best ranking endpoint: 0.331 AP, with 2 Ins in the top 5, 4 in the top 10, and 8 in the top 20. Its AP interval is 0.225–0.518. Many raw any-check likelihoods are tied, so exact top-budget membership within equal-score groups depends on the declared deterministic slug tie-break; AP retains scikit-learn's standard tie handling.

## What the more complicated models taught us

1. **The limitation is not a lack of nonlinear model capacity.** Boosting and RBF kernels do not improve corrected performance.
2. **High-dimensional semantic detail currently hurts.** Semantic Phase 1 and the 274-feature semantic combination underperform the 17-feature legacy rationale core. With only 17–19 positive outcomes, sparse semantic interactions cannot be estimated reliably.
3. **Late fusion is not automatically safer than feature concatenation.** Even correctly cross-fitted base scores and a meta-learner fail to improve AP or balanced accuracy.
4. **Ranking-specific training helps only modestly.** The pairwise model is near chance and below the raw any-check and Phase 2 endpoints.
5. **The proposed two-stage structure is not supported by these data.** It reaches 0.255 AP under strict labels but falls to 0.192 after the two sensitivity relabelings.
6. **Phase 2 is not wasted.** It remains the strongest strict learned ranker, and raw any-check is the strongest sensitivity ranker. Its value is primarily prioritization and high-recall screening—not accurate autonomous classification.
7. **Outcome definition remains consequential.** Moving only two episodes changes the ranking winner from calibrated Phase 2 to raw any-check and weakens nearly every advanced learned model.

## Recommended next step

Do not add another model family on Charles. The productive next experiment is to lock this corrected protocol and test transfer:

- apply the predeclared legacy Phase 1, Phase 2, and raw any-check endpoints to Elizabeth Yin or another VC;
- keep classification and ranking as separate tasks;
- report the raw high-recall screen alongside the more selective learned classifier;
- reserve Charles for analysis rather than further model selection;
- obtain an untouched investor or future-episode holdout before making predictive claims.

## Reproducibility artifacts

- Corrected strict metrics: `outputs/charles-expanded-calibration-2026-08-03/metrics.json`
- Corrected strict predictions: `outputs/charles-expanded-calibration-2026-08-03/predictions.csv`
- Corrected strict rankings: `outputs/charles-expanded-calibration-2026-08-03/rankings.csv`
- Conditional-interest results: `outputs/charles-expanded-calibration-2026-08-03/conditional-interest/`
- Method grids: `outputs/charles-expanded-calibration-2026-08-03/method-manifest.json`
- Cross-definition summary: `outputs/charles-expanded-calibration-2026-08-03/comparison-summary.json`
- Command: `uv run python scripts/analyze_charles_advanced_calibration.py`

