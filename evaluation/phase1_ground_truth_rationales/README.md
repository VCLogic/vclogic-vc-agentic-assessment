# Phase 1 Rationale Reference Corpus

This folder consolidates transcript-grounded rationale annotations for every
evaluation-eligible investor–pitch case. Its purpose is to evaluate whether
Phase 1 identifies the same rationale labels, directions, salience, and
investor evidence that the target VC actually expressed in the episode.

## Scientific status

These records are **automated candidate ground truth**, not final
human-validated ground truth. They are constrained to the target investor's
observable transcript statements and the audited pitch-window decision, but a
human reviewer should confirm the annotations before they are used as a final
publication benchmark. Every normalized record therefore contains:

```json
"reference_kind": "automated_transcript_observed_candidate",
"human_validated": false
```

## Structure

- `records/<vc_slug>/<episode_slug>.json` contains the common comparison schema.
- `source_artifacts/<vc_slug>/<episode_slug>.json` is an exact copy of the
  selected source annotation.
- `extraction_runs/<vc_slug>/` preserves prompts, packets, raw model events,
  normalized model outputs, validated references, and token usage for newly
  extracted gaps.
- `manifest.json` reports eligible, available, and missing counts by VC and by
  source format and source tier, plus aggregate token usage across every new
  extraction attempt and retry.

The normalized schema retains the fields required for Phase 1 comparison:
rationale label, positive/negative/neutral direction, primary/secondary
salience, confidence, activation, utterance type, decision linkage, and exact
investor evidence.

Each normalized record also records `source_tier` and `source_origin_path`.
These distinguish a newly extracted reference from prior validated work or a
legacy fallback and retain the provenance of the selected source.

## Source precedence

For each episode, the corpus builder selects the highest-quality annotation
that agrees with the current audited pitch-window label:

1. newly extracted `transcript-observed-rationales-v1` reference;
2. previously validated `transcript-observed-rationales-v1` reference;
3. older `reference_rationales` record containing taxonomy-aligned labels and
   exact evidence spans.

A richer but superseded reference whose stored decision conflicts with the
current audit cannot mask a label-consistent fallback.

## Rebuild and audit

From the LangGraph project root:

```bash
uv run python scripts/build_phase1_reference_corpus.py --allow-missing
```

Omit `--allow-missing` for the final strict build. The strict command exits
nonzero unless all 301 eligible investor–pitch cases have a valid reference.
