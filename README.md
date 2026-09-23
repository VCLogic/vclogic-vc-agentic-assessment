# VCLogic Founder Pitch Assessment Pipeline

Reusable Python engine for evidence-grounded founder pitch assessments. Phase 1 investigates investor rationales; Phase 2 synthesizes investment decisions. The package also supplies the grounded rehearsal runtime and command-line tools.

The React application and FastAPI API live in [vclogic-web-application](https://github.com/VCLogic/vclogic-web-application). This repository has no HTTP server or frontend dependency. Python imports retain the existing `vc_clone_graph` namespace.

## Setup and offline verification

Requires Python 3.11–3.13 and uv. From this checkout:

```bash
uv sync --extra dev
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run pytest -q
uv run vc-clone-graph index --config configs/fake-elizabeth-thoras.toml
uv run vc-clone-graph run --config configs/fake-elizabeth-thoras.toml
uv run vc-clone-graph verify --config configs/fake-elizabeth-thoras.toml
```

The fake-provider commands require no model service or API key. The thread limits keep numerical model tests efficient on shared machines. Historical assessment and rehearsal checks skip without the archived canonical run directory. PyTorch-dependent research checks require `uv sync --extra dev --extra personalized`; they skip when that optional dependency is absent.

## Assess a new founder pitch

```bash
uv sync --extra embeddings
uv run vc-clone-rehearsal start \
  --config configs/rehearsal-charles-v41-grounded.toml \
  --vc charles-hudson-precursor-ventures \
  --pitch /path/to/founder-pitch.txt \
  --company "Company Name" \
  --build-canonical-baseline
```

This builds the canonical assessment and starts rehearsal. The grounded configuration uses OpenRouter generation and local semantic retrieval. Set `OPENROUTER_API_KEY` in your environment and prepare the configured embedding indexes before running it. The baseline flag authorizes paid model calls. See [the framework reference](docs/FRAMEWORK_REFERENCE.md) and [rehearsal guide](docs/FOUNDER_REHEARSAL.md) for index preparation, alternate providers, frozen assessments, and session continuation. Historical examples in those references describe the original research workspace.

## View assessment rationales and decision

Run from this repository. For an existing session, one read-only command displays the initial canonical decision, activated rationales, supporting evidence IDs, controlling rationales, and diligence questions:

```bash
uv run vc-clone-rehearsal assessment \
  --config configs/investors/mac-conwell/rehearsal.toml \
  --session mac-shiftpilot-001
```

The example assumes Mac Conwell has been installed through the onboarding CLI and the `mac-shiftpilot-001` session has been run locally. Investor bundles, indexes, and session outputs are separate from this command. Use the configuration and session name from your run. Add `--format json` to export the complete investigation and decision. This command verifies the saved artifacts, makes no model calls, and works before rehearsal is finished. It shows the frozen initial assessment; `report --format markdown` shows the founder report after rehearsal finishes.

## Repository boundary

- `src/vc_clone_graph`: assessment, retrieval, validation, providers, rehearsal, and CLI.
- `configs`, `inputs`: runtime configurations, investor knowledge, audited pitches, and hash manifests.
- `evaluation`, `scripts`, `tests`: evaluation references, tooling, and engine checks.
- `outputs`, `checkpoints`, `inputs/indexes`: generated local artifacts, ignored by Git.

Run commands from this checkout so relative configuration paths resolve here. The web application installs this package and receives this checkout through `--pipeline-workspace`; it does not carry a second copy of the engine. Live output directories and historical experiment results are not migrated. Evaluation commands that consume historical runs need those artifacts supplied separately.

## Provenance

Extracted from the current working tree of `vc-digital-twins/langgraph-vc-clone-framework`, including uncommitted source changes. The original workspace is preserved. This is a new repository history.
