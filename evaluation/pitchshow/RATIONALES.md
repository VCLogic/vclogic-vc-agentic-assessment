# Transcript-Observed Rationale References

These references describe what the target VC actually expressed in the original, unredacted transcript. They are kept outside all assessment packages and shared by the founders-only and founders-and-other-vcs input modes. Extraction covers the 404 evaluation-eligible VC–episode pairs; it does not turn excluded or unavailable episodes into negative examples.

Completed: **404 of 404 eligible VC–episode references**, with zero batch errors. Final source checks verified taxonomy labels, exact target-speaker quotations, target-turn coverage, source hashes and exported records. The corpus contains 6,256 original-pitch rationale mentions and 594 later-commentary mentions. Existing references were preserved. The initial OpenRouter pass produced 164 references; the remaining 240 were completed using Codex, reusing any saved extraction stages. See [coverage by VC](RATIONALE-COVERAGE.md) and `rationale-summary.json`. These remain automated candidate references, not human-validated ground truth.

## Files

- `investors/<vc>/rationale-references/<episode>.json`: source-bound reference, exact quotes and original zero-based turn indexes.
- `investors/<vc>/rationale-references/receipts/<episode>.json`: extraction and independent verification prompts, responses and usage.
- `rationale-corpus/records/<vc>/<episode>.json`: exported records using the existing `phase1-rationale-reference-v1` comparison schema.
- `rationale-corpus/manifest.json`: selected-case coverage and missing records.
- `rationale-summary.json` and `rationale-progress.json`: extraction counts and failures. Check these for completion; this document is not a claim that every model call finished.

## Annotation Content

`rationale_mentions` preserves individual original-pitch statements. Each contains a taxonomy label, direction, salience, confidence, utterance type (question, concern, supporting argument, decision explanation, assessment), decision linkage, exact quote, speaker and original turn index.

`rationales` is the label-level comparison view, with one row per taxonomy label. When multiple mentions share a label, it selects an explicitly decision-linked statement first, then an evaluated statement over a question, then primary salience, then the latest turn. It retains matching-direction evidence and links all original mention IDs. The complete utterance history remains available; the representative view does not erase reversals or questions. Legacy-compatible values use `question`, `assessment`, or `decision_reason` and explicit/none decision linkage; `utterance_kind` and `decision_support` retain the more detailed classification.

`later_rationales` contains post-pitch interviews and later updates. `excluded_rationales` contains previews and unclear timing. Neither is included in original-pitch scoring. The original pitch includes the target's in-room decision discussion, even though that discussion was removed from the model-visible pitch. The redaction cutoff is not the end of the rationale-reference window.

## Verification and Limits

An extraction pass reads the full transcript and taxonomy. A separate model pass checks every target turn, adds omissions, removes unsupported concepts and corrects labels, timing and quotes. Deterministic checks require known taxonomy labels, complete target-turn coverage, exact attributed target-speaker quotations and valid source hashes. Unique capitalization/apostrophe differences can be repaired to the exact source spelling, with the model wording retained in the audit. Other non-verbatim quotes require an additional semantic check against the complete attributed statement: supported annotations use that exact statement as evidence, while unsupported annotations are excluded and preserved in the rejection audit. Unattributed narration inside a parsed turn cannot serve as target evidence.

These are **automated candidate references, not human-validated ground truth**. Exact quote and speaker checks do not prove perfect taxonomy interpretation or timing. Empty rationale lists mean no observed rationale was extracted, not a negative decision. The underlying reference files preserve confidence and reviewer notes for later human review.

## Resume

```bash
uv run python scripts/prepare_pitchshow_rationales.py --backend codex --workers 4
```

The Codex route uses the existing ChatGPT login: `gpt-5.6-luna` extracts and `gpt-5.6-sol` reviews through isolated `codex exec` calls. It does not call OpenRouter and remains subject to Codex usage limits. Per-call receipts record the backend and model; Codex monetary cost is unknown (the shared usage schema’s zero default is not a cost estimate). Historical OpenRouter stages can be reused when their source fingerprint matches, so some references have mixed provenance.

Use `--backend openrouter` to explicitly select the original paid API route. The backend default remains OpenRouter for existing callers. Only missing or invalidated extraction/review stages are requested. Matching completed work is reused. If the provider blocks credit usage, the process stops making new requests and reports the remaining gaps. Add `--offline` to validate and export saved reviews without any model calls; keep `--backend codex` when exporting a mixed corpus. Receipts bind transcript, target identity, decision annotation, taxonomy, model and extraction version. Use `--vc` and `--episode` for a selected case; run without those filters for the final full-coverage export. Original legacy references are preserved separately under `evaluation/phase1_ground_truth_rationales/`.

The existing Phase 1 evaluator can consume the normalized record schema, but a scoring run must use a prediction registry with the same VC slugs and case coverage. Merely creating references does not execute assessments or compute benchmark scores.
