# VCLogic Founder Pitch Assessment Pipeline

Reusable Python engine for evidence-grounded founder pitch assessments. Phase 1 investigates investor rationales; Phase 2 synthesizes investment decisions. The package also supplies the grounded rehearsal runtime and command-line tools.

The React application and FastAPI API live in [vclogic-web-application](https://github.com/VCLogic/vclogic-web-application). This repository has no HTTP server or frontend dependency. Python imports retain the existing `vc_clone_graph` namespace.

## Dependencies Across VCLogic Repositories

This pipeline is the assessment engine in the [VCLogic organization](https://github.com/VCLogic). The related repositories supply investor data or consume the engine:

| Repository | Responsibility | Relationship to This Pipeline |
|---|---|---|
| [vclogic-vc-trace-collector](https://github.com/VCLogic/vclogic-vc-trace-collector) | Resolves investor identity and collects articles, interviews, transcripts, and portfolio source pages. | Indirect upstream data source. Its investor exports are processed by investment-memory. |
| [vclogic-vc-investment-memory](https://github.com/VCLogic/vclogic-vc-investment-memory) | Analyzes collected sources and generates cited investor wikis. | Upstream data producer. Its wikis must be converted by onboarding before assessment use. |
| [vclogic-vc-inverstor-onboarding](https://github.com/VCLogic/vclogic-vc-inverstor-onboarding) | Validates wikis and prepares investor registrations, configs, retrieval indexes, and optional Pitch Show precedents. | Installs assessment-ready investor bundles into this workspace. Its CLI imports both the investment-memory validator and this engine's builders and validators. |
| [vclogic-web-application](https://github.com/VCLogic/vclogic-web-application) | Provides the React interface and FastAPI API for assessments and rehearsals. | Downstream package consumer. Imports `vclogic-vc-agentic-assessment` and uses a prepared workspace selected by `--pipeline-workspace`. |

The onboarding GitHub repository currently uses the spelling `inverstor`; its Python package and local checkout use `investor`.

### Data Flow and Package Dependencies

```text
trace-collector → investment-memory → investor-onboarding → assessment pipeline
                    cited wiki          prepared bundle          ↑
                                                           web application
                                                        invokes the engine
```

These arrows describe the data and execution flow. The Python package dependencies run as follows:

- **Onboarding imports investment-memory and assessment.** It reuses their validation and asset-building code.
- **The web application imports assessment.** It also discovers prepared onboarding bundles as files; it does not run onboarding or build missing indexes through the browser.
- **Assessment imports none of these sibling packages.** Once the required investor assets are prepared, the assessment CLI runs independently of the collector, memory, onboarding, and web processes.

### Local Checkout Layout

For local development across the projects, keep the checkouts beside one another:

```text
VCLogic/
├── vclogic-vc-trace-collector/
├── vclogic-vc-investment-memory/
├── vclogic-vc-investor-onboarding/
├── vclogic-vc-agentic-assessment/    # This repository
└── vclogic-web-application/
```

The onboarding and web projects configure editable sibling dependencies in their `pyproject.toml` files. When cloning onboarding, explicitly use the expected local directory name:

```bash
git clone https://github.com/VCLogic/vclogic-vc-inverstor-onboarding.git vclogic-vc-investor-onboarding
```

For a **new investor**, collect sources, generate the wiki, then prepare, validate, and install the onboarding bundle into this repository. Installed assets include `inputs/investors/<slug>.toml`, `inputs/wiki/<slug>/`, `inputs/indexes/`, and `configs/investors/<slug>/`, plus historical data when available. Follow each linked repository's README for its stage.

For an **already prepared investor**, use this repository's CLI directly. To use the browser interface, start the web application from its own checkout and point `--pipeline-workspace` at this assessment checkout. Investor data, generated indexes, outputs, model downloads, and credentials are separate from Python package installation.

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
