# Investor-Consistency Review (Phase 3)

An opt-in post-processing command reviews a completed agentic assessment. It does not change the existing graph or its Phase 1/2 artifacts. It makes one logical structured-generation request with Luna, using the saved pitch evidence, frozen investigation, original decision, and exact wiki/history passages already retrieved in either phase. It performs no retrieval and receives no target outcome, reference rationale labels or error stratum.

Run from the repository root:

```bash
.venv/bin/python scripts/investor_consistency_review.py \
  --run-dir /path/to/attempt-1/runs/episode-slug \
  --pitch /path/to/the/original/founders-only/pitch.txt \
  --investor 'Investor display name'
```

The default is a no-call preflight. Add `--execute` to authorize the review. Current defaults: `openai/gpt-5.6-luna`, high reasoning, 8192 output tokens. Transport retries are controlled by the existing provider. A saved summary is reused; an interrupted call with no summary is not silently retried.

## Decision contract

- `RETAIN_PHASE2`: the resolved verdict, likelihood and confidence are inherited exactly from Phase 2. The saved decision includes `decision_source: phase2`, a relative source pointer and SHA-256. Null replacement fields are canonical. Exact echoed original values are normalized to null; differing values are rejected.
- `REVISE_PHASE2`: a specific evidence-linked inconsistency must justify changing at least one of verdict, likelihood or confidence. All three replacement values are required. `verdict_changed` distinguishes a binary reversal from a score-only adjustment.

These are subjective review-adjusted scores, not statistical calibration. The reviewer can mark insufficient evidence and retain the decision. Missing information is not automatically adverse evidence. All three checks—investor constraints, historical comparisons and evidence weighting—are required. Citations must refer to supplied P/W/H IDs. This validates citation existence, not semantic correctness.

## Native artifacts

Artifacts are written under `<run-dir>/phase3/`: `input.json`, `input-provenance.json`, `prompt.txt`, `run-config.json`, `provider-response.json`, `review.json`, `decision.json`, `decision.sha256`, `validation-findings.json`, and `summary.json`. A failure keeps Phase 2 as fallback and is not a successful review. Saved excerpts sharing an evidence ID remain separately available with their source artifact path.

The three-case pilot has separate selection/scoring manifests under `evaluation/pitchshow/results/investor-consistency-pilot3-2026-10-21*`. Only those evaluation files contain actual labels and false-In/false-Out strata. The pilot used a 4096-token limit initially; one truncated case was retried once with an 8192-token limit and an identical prompt. Its original response is preserved under `phase3/attempt-1/`. Two unchanged-value serialization failures were revalidated offline, preserving their original validation findings; no extra model calls were made for those cases. This is exploratory evidence, not an overall accuracy estimate.
