# Preserved Legacy Runs and Current Results

The native artifacts remain unchanged. The catalogue normalizes only their metadata and maps the six legacy VC names to the installed investor slugs.

| Cohort | Cases | Phase 1 | Phase 2 | Input interpretation |
|---|---:|---|---|---|
| Legacy canonical v4/v4.1 | 301 | Available | Available | Mixed legacy inputs; not uniformly founders-only |
| Legacy panel-question experiment | 24 | Available | Not run | Other-investor questions added to legacy inputs |
| Current Mac/PodcastAI test | 1 | Accepted | Provisional | Strict founders-only |

## Preserved Files

- Archive: `outputs/legacy-langgraph/workspace/` (original relative layout).
- Original-input recovery: `outputs/legacy-langgraph/recovered-inputs/`.
- Copy audit: [legacy-import-manifest.json](../legacy-import-manifest.json).
- Shared result catalogue: [catalog.json](catalog.json).
- Per-VC catalogues: `../investors/<vc>/results.json`.

The archive includes native run summaries, investigations, decisions, retrieval/model records, phase hashes, old configurations, labels, rationale references, pitch inputs/audits, and the existing comparison reports. It preserves all attempts in the 24-case panel experiment. Original absolute paths inside frozen artifacts are retained as historical provenance; the catalogue supplies their new local locations. Files under `outputs/` are local artifacts ignored by Git.

## What Is Comparable

The new test produced the same summary keys, `investigation-v4.1` schema, and `decision-v4.1` schema as legacy v4.1 results. The batch runner uses the same assessment engine and creates `summary.json`, `phase1/investigation.json`, and `phase2/decision.json`, including SHA-256 sidecars and retrieval records. It refreshes this catalogue after execution.

However, the old report calls its baseline “Founder-only” while 215 of the 301 recorded inputs contain neutral investor questions. Three source input files had changed since execution; their exact original contents were recovered from saved model prompts and verified against the run hashes. All 301 original canonical inputs are now available with matching hashes. Do not relabel the complete legacy canonical cohort as the new strict founders-only mode. The legacy panel-question experiment covers only four cases per VC (24 total), and has no final investment predictions.

The newer dataset also uses different review boundaries, expanded investor memories and annotations. Common artifact formats make analysis possible, but do not make these cohorts a controlled A/B experiment. The legacy labels remain separate from the new annotations.

## Single-Pitch Test

- VC: Mac Conwell.
- Episode: `133-podcast-ai-josh-gets-displaced`.
- Mode: `founders-only`.
- Prediction: **In**. Annotated original decision: **Out**. This prediction is incorrect.
- Phase 1: accepted. Phase 2: provisional (valid frozen output, not saturated within its budget).
- Recorded provider cost: **$0.06642145**.
- Full artifact verification passed; target episode was absent from accessible precedents.
- This is an execution/format test, not evidence of overall predictive accuracy.

## Browse or Refresh

```bash
uv run python scripts/index_pitchshow_results.py
```

This creates a common result record with VC, episode, input mode/cohort, contract version, phase statuses, prediction, known in-episode decision, usage, and native artifact paths. A phase that did not run has no invented prediction. This command makes no model calls and computes no aggregate accuracy report.
