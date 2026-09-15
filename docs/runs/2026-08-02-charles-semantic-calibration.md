# Charles Hudson semantic calibration comparison

> **Superseded on 2026-08-03.** The pooled leave-one-out scores in this report are retained for audit history but must not be used as performance estimates. Removing one episode changed the training positive count from 17 to 16 exactly when the held episode was an In, allowing fold-specific hyperparameter selection and score scale to act as proxies for the held class. The corrected repeated-stratified comparison is documented in `docs/runs/2026-08-03-charles-expanded-calibration.md`.

## Executive result

This experiment tests whether the frozen outputs of the two-agent VC-clone pipeline can be converted into better pitch decisions and rankings without making any additional model calls. It covers all 86 evaluation-eligible Charles Hudson episodes and uses nested leave-one-out evaluation: each episode is scored by a model trained on the other 85 episodes, while model regularization and the classification threshold are selected using only those 85 training episodes.

Under the strict audited labels (17 Ins, 69 Outs), the strongest method is a learned calibration of Phase 2 alone. It reaches **61.1% balanced accuracy**, with **40.0% In precision**, **35.3% In recall**, and **37.5% In F1**. Its confusion counts are **6 true Ins, 9 false Ins, 60 true Outs, and 11 missed Ins**. For prioritization, it reaches **0.361 average precision**, compared with an In prevalence of 0.198, and finds **3 of 17 Ins in the top 5**, **5 in the top 10**, and **6 in the top 20**.

The main conclusion is not that Phase 2 is wasted. Phase 2 is the strongest strict-label input when it is calibrated on its own. However, concatenating the large Phase 1 semantic feature space with Phase 2 does not improve the result. The likely reason is the unfavorable ratio between the sample size and the sparse feature space: 86 observations versus 215 semantic Phase 1 features, 59 Phase 2 features, and 274 combined features.

These are development estimates, not results from an untouched holdout set.

## Methods compared

Five learned methods were evaluated:

1. **Legacy Phase 1 rationale core (17 features):** aggregate counts and confidence-weighted positive, neutral, and negative rationales by salience, plus conflicts and unanswered questions.
2. **Semantic Phase 1 (215 features):** the legacy core plus rationale-label identity, direction by label, evidence status, constraint kind and severity, severity by salience, and pitch deal-fact status.
3. **Phase 2 calibration (59 features):** the two decision heads' decisions, likelihoods and confidence, likelihood gap, ranking score, supporting/opposing/failure rationale counts, check tier, market gate, risk ledger, rationale assessments, reversal conditions, and validation status. Free-text prose is excluded.
4. **Legacy Phase 1 + Phase 2 (76 features).**
5. **Semantic Phase 1 + Phase 2 (274 features).**

For context, four unlearned references were also retained: always Out, the raw any-check decision/likelihood, the raw standard-check decision/likelihood, and the raw Phase 2 ranking score.

All learned methods use standardized, class-balanced logistic regression. The regularization strength and decision threshold are selected within each outer training fold using stratified inner cross-validation. The held-out episode's label is never used to fit its model, select regularization, or choose its threshold. The Phase 1 and Phase 2 input artifacts are hash-verified before feature extraction.

## Strict audited-label results

Population: **86 episodes: 17 Ins and 69 Outs**.

| Method | Balanced accuracy | In precision | In recall | In F1 | AP | ROC AUC | Ins in top 5 / 10 / 20 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Legacy Phase 1 | 49.1% | 19.4% | 70.6% | 30.4% | 0.279 | 0.526 | 2 / 2 / 3 |
| Semantic Phase 1 | 53.4% | 21.6% | 64.7% | 32.4% | 0.203 | 0.501 | 0 / 2 / 4 |
| **Phase 2 calibration** | **61.1%** | **40.0%** | 35.3% | **37.5%** | **0.361** | 0.445 | **3 / 5 / 6** |
| Legacy Phase 1 + Phase 2 | 51.0% | 20.7% | 35.3% | 26.1% | 0.175 | 0.384 | 0 / 0 / 5 |
| Semantic Phase 1 + Phase 2 | 42.4% | 16.1% | 52.9% | 24.7% | 0.265 | **0.665** | 0 / 0 / 3 |
| Raw any-check | 50.6% | 20.0% | **82.4%** | 32.2% | 0.269 | 0.565 | 1 / 3 / 7 |
| Raw standard-check | 50.0% | 0.0% | 0.0% | 0.0% | 0.231 | 0.564 | 1 / 2 / 4 |
| Raw ranking score | 47.8% | 19.0% | 94.1% | 31.7% | 0.186 | 0.443 | 0 / 0 / 0 |

The Phase 2 calibration improves the raw any-check from 20.0% to 40.0% In precision and reduces false positives from 56 to 9, but it also reduces In recall from 82.4% to 35.3%. It is therefore a more selective classifier, not a replacement for the high-recall raw any-check if the operational cost of missing an In is extremely high.

The strict-label Phase 2 classifier correctly identifies these six Ins:

- `39-this-pitch-is-damn-near-perfect`
- `68-dogs-dating-a-match-made-in-heaven`
- `78-got-goals-grab-a-cru`
- `102-tether-bodyguard-of-the-grid`
- `143-vital-audio`
- `148-esai`

It misses these eleven Ins:

- `77-sell-online-or-go-door-to-door`
- `83-can-small-bras-be-a-big-market`
- `104-dressd-red-carpet-or-red-ocean`
- `133-podcast-ai-josh-gets-displaced`
- `135-thoras-ai-the-twin-effect`
- `149-dopl`
- `152-oma-health`
- `164-sundae-ltk-for-groceries`
- `170-original-sunshine-the-best-bagel-youve-never-heard-of`
- `172-my-town-ai-chatgpt-meets-simcity`
- `175-above-health-the-allergy-clinic-of-the-future`

## Conditional-interest sensitivity analysis

Because episodes 41 and 127 can reasonably be treated as affirmative interest rather than strict pitch-window Ins, the entire nested evaluation was repeated after changing only those two labels to In. The sensitivity population is **19 Ins and 67 Outs**.

| Method | Balanced accuracy | In precision | In recall | In F1 | AP | ROC AUC | Ins in top 5 / 10 / 20 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Legacy Phase 1 | 54.0% | 24.5% | 63.2% | 35.3% | 0.293 | 0.515 | 2 / 3 / 4 |
| **Semantic Phase 1** | **56.2%** | **25.5%** | 73.7% | **37.8%** | 0.315 | 0.628 | 1 / 1 / 5 |
| Phase 2 calibration | 31.9% | 12.5% | 36.8% | 18.7% | **0.327** | 0.504 | **2 / 4 / 7** |
| Legacy Phase 1 + Phase 2 | 43.0% | 15.6% | 26.3% | 19.6% | 0.230 | 0.477 | 0 / 2 / 5 |
| Semantic Phase 1 + Phase 2 | 30.8% | 12.9% | 42.1% | 19.8% | 0.274 | **0.631** | 0 / 1 / 5 |
| Raw any-check | 51.8% | 22.9% | **84.2%** | 36.0% | **0.331** | 0.587 | 2 / 4 / **8** |
| Raw standard-check | 50.0% | 0.0% | 0.0% | 0.0% | 0.269 | 0.586 | 1 / 3 / 5 |
| Raw ranking score | 48.1% | 21.4% | 94.7% | 35.0% | 0.196 | 0.420 | 0 / 0 / 0 |

With these two label changes, Semantic Phase 1 becomes the strongest learned classifier, and Phase 2 remains the strongest learned ranker by AP. The raw any-check is marginally higher than calibrated Phase 2 in AP (0.331 versus 0.327) and finds one more In in the top 20. This material movement from only two disputed outcomes shows that the dataset's definition of an “In” is a substantive modeling choice, not a clerical detail.

## Interpretation

1. **Phase 2 contains useful signal.** On strict labels, its learned calibration is the best classifier and ranker of the five learned alternatives. This is a meaningful use of the decision synthesis rather than discarding it.
2. **More features are not automatically better.** Both direct Phase 1 + Phase 2 concatenations underperform the best single-phase model. With only 17–19 positive cases, the combined semantic representation is too large to estimate reliably.
3. **The best endpoint depends on the use case.** The learned Phase 2 classifier is preferable when reviewer capacity is constrained and false positives are costly. The raw any-check is preferable as a broad safety net when missing a potential In is costlier than reviewing many Outs.
4. **Classification and ranking should remain separate reported tasks.** The strict Phase 2 model has the best AP and puts five Ins in the first ten reviews, even though it misses eleven Ins at its fold-specific classification thresholds.
5. **The result is label-sensitive.** Treating conditional interest as In materially changes which representation appears strongest. Both definitions should be reported until the outcome construct is fixed prospectively.
6. **No tested Phase 1 + Phase 2 fusion is the new default.** The combination models provide no gain in balanced accuracy, AP, or early-budget hits. A future fusion attempt should reduce Phase 1 dimensionality or use a predeclared small set of Phase 1 summary scores before combining them with Phase 2.

## Reproducibility artifacts

- Strict metrics: `outputs/charles-semantic-calibration-2026-08-02/metrics.json`
- Strict episode predictions: `outputs/charles-semantic-calibration-2026-08-02/predictions.csv`
- Strict rankings: `outputs/charles-semantic-calibration-2026-08-02/rankings.csv`
- Sensitivity metrics and predictions: `outputs/charles-semantic-calibration-2026-08-02/conditional-interest/`
- Feature definitions: `outputs/charles-semantic-calibration-2026-08-02/feature-manifest.json`
- Reproduction command: `uv run python scripts/analyze_charles_calibration.py`
