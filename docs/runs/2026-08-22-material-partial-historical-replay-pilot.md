# Material-partial historical replay pilot

## Purpose

This pilot tests whether Founder Rehearsal Mode can continue a historical interaction when a recorded founder answer addresses only part of the generated investor question. Candidate historical Q&A pairs pass a cheap semantic/rationale gate and then a decision-blind Luna compatibility judge. Only exact recorded founder words classified as fully or materially partially responsive may enter the LangGraph state; unanswered clauses remain uncertainty.

The production rehearsal graph was not changed. The evaluator ran the same Charles Hudson cases as the strict replay and stored one-call baseline: episode 39 (observed In) and episode 18 (observed Out).

## Leakage controls

- Audited founder-only pitch input.
- Target episode excluded from precedent and portfolio retrieval.
- Target company aliases redacted from the wiki view.
- Actual label and decision evidence withheld from the graph and compatibility judge.
- At most two candidate answers judged per question and three accepted answers per episode.
- Historical answers retained verbatim and never supplemented.

Artifact verification passed for both sessions. Each target episode slug and its decision evidence were absent from its model-visible graph and compatibility prompts.

## Results

| Method | Episode | Actual | Predicted | Likelihood | Confidence | Accepted answers | Rationale F1 | Cost |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Material-partial replay | 39 | In | Out | 0.24 | 0.84 | 0 | 0.286 | $0.03482 |
| Material-partial replay | 18 | Out | Out | 0.15 | 0.82 | 0 | 0.333 | $0.03397 |
| Strict replay | 39 | In | Out | 0.23 | 0.84 | 0 | 0.286 | $0.04557 |
| Strict replay | 18 | Out | Out | 0.10 | 0.86 | 0 | 0.286 | $0.03738 |
| Single-call baseline | 39 | In | Out | 0.20 | 0.84 | n/a | 0.412 | $0.01855 |
| Single-call baseline | 18 | Out | Out | 0.05 | 0.93 | n/a | 0.278 | $0.01921 |

All three methods classified one of two episodes correctly and ranked the true In above the true Out. Two observations are not a performance estimate.

The material-partial run used 163,134 input tokens, 23,426 output tokens, and $0.06878. The four compatibility judgments cost $0.00199; most cost remained in the four graph calls per episode.

## Compatibility findings

### Episode 39

The graph requested fully loaded contribution margin and channel payback. Charles's observed questions and founder answers concerned customer segments, brand positioning, product targeting, and social-media growth. Both candidates were correctly classified `none`; they provided no costs, margins, or payback periods.

### Episode 18

The graph requested observed paid purchases, acquisition channels, cancellations, refunds, and continued usage. One historical answer established app-only availability and pricing; another stated aspirational engagement targets. Neither contained observed customer counts, channel evidence, cancellations, refunds, or usage. Both were correctly classified `none` rather than being accepted merely because they discussed adjacent business-model or retention topics.

## Interpretation

The compatibility layer works as intended but did not create an interactive replay on these cases. The limiting factor is upstream question fidelity: the rehearsal graph asks sensible diligence questions that the recorded conversation did not answer. Relaxing the compatibility decision would inject unsupported information and invalidate the evaluation.

This result separates two claims. The system can generate useful founder-diligence questions, but these two questions did not reproduce Charles's observed questioning closely enough for counterfactual historical replay. Historical replay therefore measures behavioral fidelity and answer availability, while live founder rehearsal measures whether the generated questions are useful when a real founder can answer them.

## Artifacts

- `outputs/rehearsal-material-partial-pilot-2026-08-22/pilot-summary.json`
- `outputs/rehearsal-material-partial-pilot-2026-08-22/charles-hudson-precursor-ventures/material-partial-39-this-pitch-is-damn-near-perfect-0ada395b/`
- `outputs/rehearsal-material-partial-pilot-2026-08-22/charles-hudson-precursor-ventures/material-partial-18-rowvigor-1a7d346d/`
