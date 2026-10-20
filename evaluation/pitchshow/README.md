# Pitch Show Evaluation Dataset

Per-investor transcripts, source-linked investment annotations, and redacted pitch inputs for the 19 current investor memories. This directory is outside the assessment engine's `inputs/` tree: original transcripts, labels, and review records must not be supplied to the assessed model.

## Layout

```text
pitchshow/
  collection-summary.json          # Collection coverage and missing transcripts
  summary.json                     # Counts and eligibility by investor
  reviews/<episode-slug>.json      # Shared extraction and independent verification
  investors/<vc-slug>/
    source/profile.json           # Official investor profile and reported investments
    source/episodes/<slug>.json    # Original parsed episode metadata and transcript
    source/collection.json        # Crawl coverage, sources, failures, missing text
    transcripts/<slug>.txt         # Original available transcript, unchanged
    annotations.json              # In-episode decisions and separate later outcomes
    pitches/pitch-<id>.txt         # Model-visible, neutral-named redacted inputs
    audits/pitch-<id>.json         # Boundaries, transformations, hashes and review links
    summary.json
```

Original episode slugs follow the original LangGraph/scraper naming convention. Redacted pitch filenames use stable opaque IDs because an episode title can itself reveal an outcome. `annotations.json` provides the mapping. The same source episode is saved in each relevant VC folder, but model review is performed once across its target investors.

## Decision Labels

- `pitch_window_decision`: `In`, `Out`, or `Unobserved`. This is the evaluation target: the target VC's final stated decision during the original pitch. An explicit conditional check commitment can count as `In`; its condition is retained.
- `initial_response` and `decision_context`: distinguish an initial response, a reversal in the same session, later diligence, off-panel participation, and unclear timing.
- `later_decision`: `In`, `Out`, or `Unknown`, requiring separately verified transcript evidence about a later outcome. A pitch commitment is not proof that money subsequently changed hands.
- `reported_profile_investment`: an investment reported on the official investor profile, or `Unknown`. An absent investment listing never becomes an `Out`.
- `pitch_evidence` and `later_evidence`: verbatim quotes with zero-based speaker-turn indexes into the source transcript.
- `evaluation_eligible`: true only when the in-episode decision evidence and the redacted pitch pass the gates. All unavailable, disputed, unsafe, and non-pitch examples remain in the annotations without a published pitch.

Reviews are **machine-generated and independently model-verified, not human-audited ground truth**. Earlier audited labels from this repository are retained as `prior_audit`; disagreements with observed new labels are withheld from evaluation pending reconciliation. Earlier label files remain unchanged. For this dataset, score `pitch_window_decision`, not a later corrected funding outcome.

## Pitch Redaction

The workflow reuses `parse_speaker_turns` and `build_pitch_only_document` from the original LangGraph assessment implementation. It adds review of the start and end boundaries and independent semantic review:

1. Skip teasers, introductions and retrospective segments.
2. End before the **first panel decision or outcome-revealing negotiation**, including another investor's decision.
3. Retain source-grounded founder statements and neutral questions; remove investor evaluations, host narration and unattributed continuation lines through the existing renderer. Preserve factual uses of names in founder statements (for example, Apple Mac and Purely Elizabeth); the semantic review handles actual investor-specific clues.
4. Review every retained line for implicit clues, including thanks, celebrations, investor-specific negotiations, and reactions. Remove flagged lines verbatim rather than rewriting business facts.
5. Verify source speaker, quote, boundaries, target coverage and output hashes. Withhold unsafe inputs.

The common early cutoff intentionally loses some later factual Q&A. No automated process can guarantee that every subtle outcome clue has been removed; inspect the audits before treating the dataset as a human-reviewed benchmark.

## Reproduce or Resume

Run from the assessment repository. Collection reuses the sibling onboarding project's official-site collector; the assessment package itself gains no dependency on onboarding.

```bash
uv run --with beautifulsoup4 python scripts/pitchshow_evaluation_sources.py \
  --onboarding-workspace ../vclogic-vc-investor-onboarding \
  --memory-workspace ../vclogic-vc-investment-memory

# Uses OPENROUTER_API_KEY from the environment or this checkout's .env.
# Makes paid model calls only for transcripts without matching saved reviews.
uv run python scripts/prepare_pitchshow_evaluation.py --workers 8

# Rebuild per-investor annotations, redacted pitches and summaries without model calls.
uv run python scripts/prepare_pitchshow_evaluation.py --publish-only
```

This collection targets the 19 October 17 onboarding bundles. Existing collection folders and cached downloads are reused on resume; choose a new `--output` directory and matching review `--root` for a fresh collection. Reviews are reusable only when the transcript, panel metadata, targets, model, and review version match. Download failures and missing transcripts are recorded rather than silently converted into negative labels. Raw HTML caches are kept locally and ignored by Git.

## Use for Assessment

Only eligible files under `investors/<vc>/pitches/` are pitch inputs. Keep the matching annotation and audit files outside the inference context. Before scoring historical assessments, exclude the held-out episode and company from wiki and precedent retrieval as in the original pipeline's firewall; redacting the pitch alone does not prevent historical retrieval leakage. No assessment or benchmark score is produced by this preparation workflow.

## Target-Specific Input Modes

The preferred evaluation inputs are now organized by **target VC and original episode slug**:

```text
investors/<vc-slug>/cases/<episode-slug>/
  founders-only/
    pitch.txt
    audit.json
  founders-and-other-vcs/
    pitch.txt
    audit.json
```

- **founders-only:** retained founder statements and answers only.
- **founders-and-other-vcs:** the same founder material plus retained neutral questions from other investors. Every target-VC turn is removed using original speaker attribution, before anonymized question labels are considered. Other investors' evaluations, decisions, and negotiations are not restored.

Both modes are subsets of the existing semantically reviewed pitch. They never restore removed lines. Founder answers to omitted questions remain, so the founders-only text can have less conversational context. The audit records source turn indexes, retained speakers, and hashes. Only the 404 eligible VC–episode pairs are published (808 inputs); missing or disputed cases remain in `annotations.json`.

Regenerate after rebuilding the reviewed dataset:

```bash
uv run python scripts/prepare_pitchshow_variants.py
```

Prepare a verified assessment package, without model calls:

```bash
uv run python scripts/prepare_pitchshow_variants.py \
  --vc mac-conwell \
  --episode 133-podcast-ai-josh-gets-displaced \
  --mode founders-only \
  --company PodcastAI --company 'Podcast AI'
```

Change `--mode` to `founders-and-other-vcs` for the second input. Add **`--run`** to execute the paid assessment with the installed VC's canonical configuration. No TOML files need to be created manually. Packages, saved run configurations, and outputs go under `outputs/pitchshow-evaluation/<vc>/<episode>/<mode>/` with a pitch-and-alias hash subfolder.

Company aliases are loaded automatically from `company-aliases.json` when `--company` is omitted. You may still supply explicit aliases for an individual package. The command preserves the original episode slug in the run configuration and binds aliases into the verified package. The existing assessment runtime excludes that episode from precedent retrieval and filters related company references from precedents and wiki sections. Merely supplying a `pitch.txt` to an unrelated live-pitch command does not establish this historical exclusion.

Episode slugs are used for organization and exclusion metadata, not inserted into pitch text. The older `pitches/pitch-<id>.txt` files remain the shared, panel-question version and **can contain the target VC's questions**; use the new `cases/` inputs for these two modes. Labels and source reviews stay outside the selected pitch input. The preparation command verifies packaging; it does not claim that an assessment has run or that model predictions are correct.


## Company Aliases and Resumable Batch Execution

`company-aliases.json` covers all **132 unique eligible episodes** used by the **404 VC–episode pairs**. Each entry contains the company names used for exclusion, exact supporting source excerpts, and the source-file hash. Names were checked against the downloaded episodes, including spelling variants and explicit name changes (for example, Your Beet/Sundae). This is agent-reviewed source verification, not human sign-off or a guarantee of every possible historical alias. Generic company names can conservatively exclude unrelated text as well. The mapping and its source excerpts are never supplied as pitch content to the assessment model.

From the repository root:

```bash
# Validate the complete selection without packaging or assessment calls (default action).
uv run python scripts/run_pitchshow_batch.py --mode both

# Exercise packaging and retrieval preflight for a small selection, without paid generation.
uv run python scripts/run_pitchshow_batch.py \
  --vc mac-conwell --episode 133-podcast-ai-josh-gets-displaced \
  --mode both --preflight

# Run all 404 founders-only assessments. This makes paid model calls.
uv run python scripts/run_pitchshow_batch.py --mode founders-only --execute

# Or run both modes: 808 assessments.
uv run python scripts/run_pitchshow_batch.py --mode both --execute
```

Use repeated `--vc` or `--episode` flags to select cases. `--limit N` selects the first N jobs, including already completed jobs; it is useful for a small initial run. The default mode is `founders-only`. Installed VC configurations/indexes and the configured API credentials are required for execution. Preflight may load or download the configured local embedding model; it does not call a paid generation provider.

### Assessments Through Codex

Add `--backend codex` to use the local Codex CLI's ChatGPT login instead of the configured API. The runner keeps the configured generation models (removing the OpenRouter `openai/` prefix), phase settings, local embeddings and exclusion checks. Codex usage limits apply. Provider/model provenance is recorded; monetary cost is unknown, and the shared usage schema's zero cost value must not be interpreted as a measured free run. Backend overrides receive separate job identities and do not change installed canonical templates. Codex CLI cannot enforce the API-style per-call output-token limit; receipts record that limitation. If constrained structured output fails with `max_output_tokens`, the adapter retries once with the unchanged schema in the prompt and validates the resulting JSON locally against that schema. Other errors are not retried through this fallback.

For one annotated In (Finnecto) and one Out (Aura Finance) from Jenny, each in both input variants:

```bash
uv run python scripts/run_pitchshow_batch.py \
  --vc jenny-fielding-everywhere-ventures \
  --episode 168-finnecto-where-deals-dont-die \
  --episode 159-aura-finance-chasing-the-ghost-of-mint \
  --mode both --backend codex --execute
```

Re-run the same execution command to continue a batch. The runner operates sequentially and locks the batch directory against concurrent execution. Completed assessments are skipped only after their artifacts and saved summary hash pass verification. Failed or interrupted cases get a fresh numbered attempt on the next invocation; earlier attempts and logs remain available. A result that fails verification is recorded as failed, not silently accepted. Each invocation makes at most one new attempt per selected case.

Execution identities bind the pitch, reviewed source, aliases, installed asset fingerprints, canonical configuration, and engine/runner code. Changing these creates new jobs rather than treating old predictions as current. A change during execution stops that case before packaging; rerun the batch to build a current plan.

```text
outputs/pitchshow-evaluation/
  batch/
    plan-<hash>.json                  # Label-free selected jobs
    jobs/<job-id>/status.json         # Completion/failure and attempt history
    jobs/<job-id>/attempt-<n>.log      # Assessment execution log
    latest-summary.json              # Counts for the last invocation
  <vc>/<episode>/<mode>/<job-id>/
    inputs/                          # Verified package with exclusion metadata
    attempt-<n>/
      run-config.json
      preflight.json
      checkpoints.sqlite
      runs/<episode>/
        summary.json
        phase1/investigation.json
        phase2/decision.json
```

Status records point to the verified result summary and retain usage data when available. These artifacts are ready for a later comparison with `annotations.json`; this batch runner does **not** compute accuracy or other evaluation scores. The known in-episode decisions remain outside the model's pitch input and are not used to order jobs.


## Preserved Legacy Results

The six-investor legacy archive and the completed new smoke test are indexed in [results/README.md](results/README.md). This explains the legacy input-mode differences and Phase-1-only panel-question coverage. New batches refresh the common and per-VC catalogues automatically after execution.


## Rationale Detection References

See [transcript-observed rationale references](RATIONALES.md) for target-VC taxonomy annotations, exact original turns, timing separation, and the shared A/B scoring reference. These files are never copied into assessment inputs.
