# Founder Rehearsal Mode

Founder Rehearsal Mode is an interactive LangGraph workflow that lets a founder test a pitch against one of the six source-linked VC profiles in this repository. It asks one investor-specific question at a time, records founder answers verbatim, updates an auditable rationale map, and produces a simulated pitch-stage In or Out assessment plus founder coaching.

The output is a simulation generated from public evidence. It is not the real investor, is not endorsed by the investor, and must not be represented as private knowledge or investment advice.

## Product boundary

The project now has two deliberately separate command paths:

- `vc-clone-graph` remains the retrospective, fixed-input scientific evaluation workflow.
- `vc-clone-rehearsal` is the interactive founder-support workflow.

Founder answers never enter the scientific pitch-only artifacts. The two workflows share only stable infrastructure: investor wikis, rationale taxonomy, local embeddings, historical precedent retrieval, portfolio memory, model providers, and evidence-validation conventions.

## Supported investors

The CLI discovers investors from `inputs/investors/*.toml`; there are no VC-specific branches in the rehearsal graph.

```bash
uv run vc-clone-rehearsal investors \
  --config configs/rehearsal/openrouter-luna.toml
```

The current registry includes Charles Hudson, Cyan Banister, Elizabeth Yin, Jesse Middleton, Jillian Manus, and Phil Nadel.

## Configuration

The reusable configuration is:

```text
configs/rehearsal/openrouter-luna.toml
```

It configures:

- OpenRouter generation with `openai/gpt-5.6-luna`;
- a maximum of eight founder questions;
- one bounded schema-repair attempt;
- 8,192 output tokens per structured call;
- pinned local `nomic-ai/nomic-embed-text-v1.5` sentence-transformer embeddings;
- hybrid wiki retrieval;
- semantic historical-precedent retrieval;
- and portfolio-memory retrieval.

The configuration contains only the environment-variable name `OPENROUTER_API_KEY`, never the credential value. Place the actual credential in `.env` or the process environment.

## Start a session

Pass either a plain-text file:

```bash
uv run vc-clone-rehearsal start \
  --config configs/rehearsal/openrouter-luna.toml \
  --vc charles-hudson-precursor-ventures \
  --pitch founder-pitch.txt
```

or paste the pitch as an argument:

```bash
uv run vc-clone-rehearsal start \
  --config configs/rehearsal/openrouter-luna.toml \
  --vc elizabeth-yin-hustle-fund \
  --text "We help ..."
```

If the company name may already occur in the investor memory, provide one or more aliases so the ephemeral wiki view redacts them:

```bash
uv run vc-clone-rehearsal start \
  --config configs/rehearsal/openrouter-luna.toml \
  --vc charles-hudson-precursor-ventures \
  --pitch founder-pitch.txt \
  --company "Example Company" \
  --company "ExampleCo"
```

The command returns a session ID and one public question. During the interview it exposes the neutral investment dimension being explored, but it withholds the evolving decision, likelihood, rationale direction, salience, and confidence.

## Answer, finish, or retry

```bash
uv run vc-clone-rehearsal answer \
  --config configs/rehearsal/openrouter-luna.toml \
  --session SESSION_ID \
  --text "Our month-three retention is 70%."
```

The answer is stored verbatim as an unverified founder claim. It does not modify the immutable pitch.

The founder may finish early:

```bash
uv run vc-clone-rehearsal finish \
  --config configs/rehearsal/openrouter-luna.toml \
  --session SESSION_ID
```

If a structured model response remains invalid after its one repair attempt, previously accepted turns remain intact. Retry the failed graph node without repeating the founder answer:

```bash
uv run vc-clone-rehearsal retry \
  --config configs/rehearsal/openrouter-luna.toml \
  --session SESSION_ID
```

The interview otherwise stops adaptively when another answer is unlikely to change or materially strengthen the assessment, or after eight accepted answers.

## Report and verification

```bash
uv run vc-clone-rehearsal report \
  --config configs/rehearsal/openrouter-luna.toml \
  --session SESSION_ID \
  --format markdown

uv run vc-clone-rehearsal verify \
  --config configs/rehearsal/openrouter-luna.toml \
  --session SESSION_ID
```

The completed report contains:

- initial and final simulated In or Out assessments;
- likelihood and confidence;
- source-linked rationale states;
- unresolved uncertainties and reversal conditions;
- founder answers that materially changed the assessment;
- investor-specific pitch improvements;
- reflection prompts;
- and token/cost usage.

Simulated judgment and founder coaching are shown as separate report sections.

## Compare independent VC sessions

Run the same immutable pitch independently against two or more VCs, then compare completed sessions:

```bash
uv run vc-clone-rehearsal compare \
  --config configs/rehearsal/openrouter-luna.toml \
  --session CHARLES_SESSION \
  --session YIN_SESSION \
  --format csv
```

Comparison is deterministic and reads verified completed artifacts. It refuses sessions with different pitch hashes, never invokes another model, and never allows one VC's evolving state to influence another VC.

## Evidence and retrieval

Stable evidence prefixes indicate provenance:

- `P-*` — immutable original-pitch evidence;
- `A-*` — verbatim founder answers;
- `W-*` — exact investor-wiki sections;
- `H-*` — semantically retrieved historical precedent excerpts;
- `PE-*` — portfolio entities linked to exact earlier disclosure records.

The graph revisits retrieval on the initial assessment and before each new question. Local Nomic embeddings and lexical retrieval jointly find relevant wiki and precedent evidence. The model receives exact source IDs and taxonomy definitions, and material rationale conclusions must use valid taxonomy labels and existing evidence IDs.

## Artifact layout

```text
outputs/rehearsals/<vc-slug>/<session-id>/
├── pitch.txt                       # immutable founder input
├── manifest.json                   # session identity and pitch digest
├── session-config.json             # model/retrieval provenance, no secrets
├── artifact-index.json             # hashes of accepted and raw artifacts
├── state.json                      # latest serializable graph state
├── initial-assessment.json
├── final-assessment.json
├── founder-report.json
├── summary.json
├── usage.json
├── turns/
│   └── turn-XX/
│       ├── question.json
│       ├── answer.json             # verbatim founder claim
│       └── update.json
├── calls/
│   └── <phase>/call-XXX.json       # exact prompt, schema, output, usage
└── findings/                       # resumable structured-output failures
```

SQLite checkpoints are kept at `outputs/rehearsals/checkpoints.sqlite`. They allow each CLI invocation to resume the exact LangGraph thread after process interruption.

## Historical canaries

The canary module supports development evaluation against historical episodes. It starts from the audited founder-only pitch cut, excludes the target episode from precedent retrieval, extracts only the target VC's observed questions and the associated founder answers, and excludes other investors' commentary and the target VC's actual decision from model inputs.

Available diagnostics include question semantic similarity, question-rationale overlap, rationale precision/recall/F1, direction and salience accuracy, likelihood information gain, final decision correctness, schema repairs, calls, latency, token use, and USD cost. These are development diagnostics unless evaluated on a separately locked set.
