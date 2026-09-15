# LangGraph VC Clone Framework

Reusable, evidence-grounded prototype for modeling an individual venture
capitalist's pitch evaluation. It preserves the established methodology:

1. **Rationale Investigation** asks investor-like questions, searches and reads
   the investor wiki iteratively, and freezes evidence-linked rationales without
   producing a verdict.
2. **Investment Decision Synthesis** receives only the frozen investigation,
   weighs the rationales, tests the countercase, and freezes two explicit
   recommendations: whether the investor would write any check, including a
   bounded exploratory check, and whether the opportunity clears the stricter
   standard-check bar.

For compatibility, the top-level `decision`, `investment_likelihood`, and
`decision_confidence` always equal the **any-check** endpoint. The separate
`standard_check` endpoint evaluates a normal institutional commitment.
`recommended_check_tier` reconciles the two, while `ranking_score` remains an
independent priority for human review rather than another binary threshold.

LangGraph controls bounded loops and SQLite checkpoints. Pydantic validates
outputs. The inference firewall admits only an audited pitch, the selected
investor wiki, registry, taxonomy, and hash manifest. Labels and complete
transcripts are not present.

## Setup

```bash
cd /home/dpasch01/vc-digital-twins/langgraph-vc-clone-framework
uv sync --extra dev
uv run pytest -q
```

## Deterministic local verification

This path makes no model or API call:

```bash
uv run vc-clone-graph index --config configs/fake-elizabeth-thoras.toml
uv run vc-clone-graph run --config configs/fake-elizabeth-thoras.toml
uv run vc-clone-graph verify --config configs/fake-elizabeth-thoras.toml
```

## Live founder rehearsal from a new pitch

The grounded rehearsal command can build the required canonical Phase 1 and
Phase 2 assessment and immediately begin the interactive LangGraph interview:

```bash
uv run vc-clone-rehearsal start \
  --config configs/rehearsal-charles-v41-grounded.toml \
  --vc charles-hudson-precursor-ventures \
  --pitch /path/to/founder-pitch.txt \
  --company "Company Name" \
  --build-canonical-baseline
```

`--build-canonical-baseline` is explicit authorization for the paid OpenRouter
Phase 1 and Phase 2 calls. The command creates a session-owned, manifest-bound
pitch package; runs the canonical v4.1 workflow with local semantic precedent
retrieval and portfolio-memory checks; verifies the frozen outputs; and uses
them as the immutable starting point for rehearsal. It then returns either the
first investor-like question or a completed report.

Continue the same session without rebuilding or repaying for the baseline:

```bash
uv run vc-clone-rehearsal answer \
  --config configs/rehearsal-charles-v41-grounded.toml \
  --session SESSION_ID \
  --text "The founder's answer"
```

An interrupted canonical build can be resumed by repeating `start` with the
same `--session`, pitch, company alias, and `--build-canonical-baseline`. To
reuse a complete canonical assessment, replace the build flag with
`--canonical-baseline PATH`. The supplied run must match the selected VC and
the pitch's exact SHA-256 digest. The two baseline flags are mutually exclusive.

At least one `--company` alias is required so the wiki, precedent, and portfolio
retrievers can exclude target-company leakage. Provider tokens and reported
cost are retained in the canonical bootstrap audit. API keys and private model
reasoning are not stored.

## Ollama prototype

```bash
ollama pull qwen3:0.6b
ollama pull nomic-embed-text
uv run vc-clone-graph index --config configs/ollama-elizabeth-thoras.toml
uv run vc-clone-graph run --config configs/ollama-elizabeth-thoras.toml
uv run vc-clone-graph verify --config configs/ollama-elizabeth-thoras.toml
```

This is an **operational smoke-test configuration**, not a research-quality VC
model. It uses JSON mode plus strict graph-side Pydantic validation because this
host's Ollama JSON-Schema decoder is exceptionally slow. Retrieved wiki sections
remain exact-read and hash-bound. See
[`docs/OLLAMA_CAPABILITY_REPORT.md`](docs/OLLAMA_CAPABILITY_REPORT.md) for the
measured T4 results and the current runtime blocker.

## OpenAI production provider

Copy `.env.example` to `.env`, set `OPENAI_API_KEY`, and use the supplied
`configs/openai-elizabeth-thoras.toml` (changing the model if required):

```toml
[provider]
kind = "openai"
model = "YOUR_APPROVED_OPENAI_MODEL"
embedding_model = "text-embedding-3-small"
```

The `.env` file is ignored. Keys are not serialized into graph state, prompts,
checkpoints, model metadata, or run artifacts.

```bash
uv run vc-clone-graph index --config configs/openai-elizabeth-thoras.toml
uv run vc-clone-graph run --config configs/openai-elizabeth-thoras.toml
uv run vc-clone-graph verify --config configs/openai-elizabeth-thoras.toml
```

## OpenRouter generation with local retrieval

OpenRouter can provide Phase 1 and Phase 2 generation while the cached local
Nomic index continues to provide wiki retrieval. Put `OPENROUTER_API_KEY` in
`.env`; OpenRouter billing is separate from Codex or ChatGPT usage.

The Elizabeth Yin Luna example requires the local Ollama server and
`nomic-embed-text`, but it does not rebuild the saved wiki index:

```bash
ollama pull nomic-embed-text
uv run vc-clone-graph run \
  --config configs/openrouter-luna-elizabeth-thoras.toml
uv run vc-clone-graph verify \
  --config configs/openrouter-luna-elizabeth-thoras.toml
```

The example uses `openai/gpt-5.6-luna`, requires schema-capable OpenRouter
routing, denies provider data collection where compatible routes are available,
and records provider-reported token usage and cost in the run artifacts.
Set a fresh `output_root` and `checkpoint_path` in the TOML before starting a
new independent attempt.

The Charles comparison package shares one wiki index across three episode-scoped
manifests. Build the index once, then run and verify each episode sequentially:

```bash
uv run vc-clone-graph index \
  --config configs/openrouter-luna-charles-harper-wilde.toml
uv run vc-clone-graph run --config configs/openrouter-luna-charles-amazon.toml
uv run vc-clone-graph run --config configs/openrouter-luna-charles-nectir.toml
uv run vc-clone-graph run --config configs/openrouter-luna-charles-harper-wilde.toml
```

Use `vc-clone-graph verify` with each corresponding config after the runs.

### Charles full-precedent canary

The canonical Charles project is
`inputs/data/investors/charles-hudson-precursor-ventures/precedents`. Rebuild it
from the audited 113-episode decision ledger, then rebuild manifests only for
pitch cuts that already exist:

```bash
uv run python -m vc_clone_graph.precedent_builder \
  --transcripts ../data/episodes \
  --ledger evaluation/labels/charles_pitch_window_decisions.json \
  --output inputs/data/investors/charles-hudson-precursor-ventures/precedents \
  --investor-alias Charles \
  --investor-alias 'Charles Hudson'

for pitch in inputs/data/investors/charles-hudson-precursor-ventures/pitches/*.txt; do
  slug=${pitch##*/}
  uv run python -m vc_clone_graph.manifest_builder \
    --root inputs \
    --vc charles-hudson-precursor-ventures \
    --episode "${slug%.txt}"
done
```

The canary operator sequence is index, preflight, run or resume, optional
decision replay, then verify:

```bash
uv run vc-clone-graph index \
  --config configs/openrouter-luna-charles-full-precedent-canary.toml
uv run vc-clone-graph preflight \
  --config configs/openrouter-luna-charles-full-precedent-canary.toml
uv run vc-clone-graph run \
  --config configs/openrouter-luna-charles-full-precedent-canary.toml
# If an interrupted run has a usable checkpoint:
uv run vc-clone-graph resume \
  --config configs/openrouter-luna-charles-full-precedent-canary.toml
# With a fresh replay config/output/checkpoint, replay only the frozen decision phase:
uv run vc-clone-graph decide --config path/to/fresh-replay.toml \
  --phase1-from outputs/openrouter-luna-charles-full-precedent-canary/20-harper-wilde/phase1
uv run vc-clone-graph verify \
  --config configs/openrouter-luna-charles-full-precedent-canary.toml
```

`preflight` never instantiates a generation provider. It verifies the audited
pitch package and saved wiki/precedent indexes, filters the target, and reports
the pitch hash, corpus/access counts, filtered-manifest hash, models, and
budgets as JSON. If either index is absent it reports `not_ready` and the exact
index command. Indexing uses local Ollama `nomic-embed-text` when available and
persists a valid lexical fallback when embeddings fail; it does not call
OpenRouter.

LangGraph governs both bounded phases. OpenRouter supplies Luna generation,
while the canonical corpus gives both phases direct, hash-verified access to
every non-target transcript and decision. `top_k` governs search navigation,
not corpus visibility. The target episode is removed before access and remains
available only through its audited pitch cut. An `unobserved` decision means no
auditable decision was found; it must not be treated as `Out`.

Run artifacts expose a public process trace—retrieval calls, exact evidence
reads, validation events, and structured decision explanations—not hidden
chain-of-thought. Usage artifacts account for tokens and provider-reported cost
by phase. OpenRouter billing is separate from Codex or ChatGPT usage. Treat the
single Harper Wilde run as a canary gate: review its boundary checks, evidence,
trace, token use, and cost before authorizing any batch run.

### Natural adaptive v4 workflow

Contract v4 keeps the same two research phases but removes the model-facing
episode inventory and the old any-check/standard-check split. The rationale
investigator receives the exact pitch, all 44 rationale labels with their full
definitions and parent dimensions, and only the exact wiki/precedent passages
opened for its current questions. Investment Decision Synthesis then returns
one `In` or `Out` capital-commitment decision, one likelihood, one confidence,
a plain-language evidence-linked justification, and an independent
`review_priority_score` for ranking pitches.

Each phase performs a planning call followed by a structured generation call.
The model may stop after one iteration or request another focused search, up to
four iterations per phase. A malformed planner response is retried once with a
concise repair prompt before the deterministic fallback is used; the Charles v4
configuration allows 8,192 output tokens for each planning attempt. Initial and
repair attempts are stored and charged separately. A valid current result is
retained even when a later reconsideration is malformed or the iteration cap is
reached. Consequently a normal sufficient run uses four generation calls. The
bounded maximum is 24 when every planner needs its one repair attempt, plus one
optional post-decision taxonomy-reflection call when Phase 1 records a material
unmapped observation. Reflection only creates a human-review proposal and never
edits the canonical taxonomy.

Run Charles episode 18 with the reusable local Nomic/OpenRouter configuration:

```bash
uv run vc-clone-graph preflight \
  --config configs/openrouter-luna-charles-v4.toml
uv run vc-clone-graph run \
  --config configs/openrouter-luna-charles-v4.toml
# Resume only after an interrupted run, using the same checkpoint:
uv run vc-clone-graph resume \
  --config configs/openrouter-luna-charles-v4.toml
uv run vc-clone-graph verify \
  --config configs/openrouter-luna-charles-v4.toml
```

Change `episode_slug`, `output_root`, and `checkpoint_path` for a different
pitch or a fresh independent attempt. The target episode is excluded inside
the precedent runtime before semantic retrieval; prompts contain opened exact
source text, not registry rows, hashes, or search-history bookkeeping.

### Replay only Investment Decision Synthesis

To test a revised Phase 2 method without paying to regenerate Phase 1, point a
fresh run configuration at a frozen Phase 1 directory:

```bash
uv run vc-clone-graph decide \
  --config configs/openrouter-luna-charles-nectir-deliberation.toml \
  --phase1-from outputs/openrouter-luna-charles-comparison/127-nectir-the-classroom-of-the-future/phase1
```

`decide` verifies `investigation.json` against its SHA-256 sidecar, copies it
into a fresh run root, records source provenance, and runs only Phase 2. It does
not load the wiki index, retrieve evidence, expose the pitch, or call a Phase 1
model. The Phase 2 output includes an observable audit trail: every frozen
rationale is assessed exactly once and 3–8 ordered deliberation steps record
questions, rationale references, check-tier implications, and continuous
likelihood changes. This is a structured decision explanation, not hidden model
chain-of-thought.

## Add an investor or pitch

Use the importer against an inference package whose pitch and wiki have already
been audited and manifest-bound:

```bash
uv run python -m vc_clone_graph.importer \
  --source /path/to/source-framework \
  --destination /path/to/new-input-package \
  --vc investor-slug \
  --episode episode-slug
```

One graph implementation serves every investor. Investor identity, wiki,
check-tier vocabulary, models, iteration bounds, and retrieval limits are data
and configuration rather than investor-specific Python code.
Packages containing several pitches use
`data/investors/<vc>/manifests/<episode>.json`; legacy single-episode
`source-manifest.json` packages remain supported.

## Outputs

Each run preserves retrieval calls, exact reads, model responses, usage,
validation events, SQLite checkpoints, canonical frozen JSON, SHA-256 sidecars,
and a summary. A valid but unsaturated final Phase 1 candidate is explicitly
marked provisional; a phase with no valid candidate is marked failed.

## Canonical evaluation

The canonical registry in `evaluation/canonical_runs.json` identifies the
definitive audited labels and inference artifacts for every investor. Recovery
sources have explicit precedence over earlier failed runs, so evaluation does
not depend on directory-name scanning or manual deduplication.

List the registered investors:

```bash
uv run python scripts/evaluate_vcs.py --list-vcs
```

Print classification and ranking tables for one investor:

```bash
uv run python scripts/evaluate_vcs.py \
  --vc charles-hudson \
  --top-k 1,3,5,10,20
```

`--vc` is repeatable. To evaluate all investors and write both the Markdown
report and the three CSV datasets:

```bash
uv run python scripts/evaluate_vcs.py \
  --all \
  --top-k 1,3,5,10,20 \
  --format both \
  --output-dir reports/evaluation/canonical-all-vc
```

Classification uses the Phase 2 `decision`; ranking uses Phase 2
`investment_likelihood`. Ranking is interpreted within investor. The all-VC
output includes macro averages and a clearly labeled pooled diagnostic, but it
does not assume that different investors' likelihood scales are calibrated
against one another.

## Phase 1 rationale references

Transcript-observed rationale references for the canonical evaluation cases
live in `evaluation/phase1_ground_truth_rationales/`. The corpus combines prior
validated work with newly extracted gaps, preserves the selected source
artifact, and normalizes both rich and legacy annotations to one comparison
schema. It deliberately labels the records as automated candidate ground truth
until a human reviewer confirms them.

Rebuild the corpus and fail if any canonical case is missing:

```bash
uv run python scripts/build_phase1_reference_corpus.py
```

Use `--allow-missing` during an in-progress extraction audit. The corpus
manifest reports coverage by investor, source format, and provenance tier.

Evaluate the frozen canonical Phase 1 outputs against that corpus:

```bash
uv sync --extra embeddings
uv run python scripts/evaluate_phase1_rationales.py \
  --references evaluation/phase1_ground_truth_rationales \
  --taxonomy ../agentic-vc-clone-framework/taxonomy/codebook_v_final.json \
  --embedding-model nomic-ai/nomic-embed-text-v1.5 \
  --embedding-revision e9b6763023c676ca8431644204f50c2b100d9aab \
  --output reports/evaluation/phase1-rationale-all-vc
```

Phase 1 defaults to the frozen hybrid v4/v4.1 registry at
`evaluation/canonical_runs_v4_v41_portfolio_2026-08-15.json`. Contract v4.2 is
retained only as an experimental rationale-lock audit and is not a canonical
Phase 1 source.

Use repeatable `--vc <slug>` arguments for a subset, or `--skip-semantic` for
exact-label metrics without local embeddings. The evaluator never reruns an
investor model or uses Phase 2 as a rationale signal. It writes Markdown, CSV,
and a hash-bound manifest; semantic similarity is explicitly diagnostic and
does not modify exact taxonomy-label scores.

## Web application

The frontend, FastAPI API, and web launch instructions are maintained in [vclogic-web-application](https://github.com/VCLogic/vclogic-web-application).
