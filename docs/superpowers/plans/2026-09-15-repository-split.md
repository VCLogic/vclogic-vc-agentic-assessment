# Repository Split Implementation Plan

**Goal:** Extract the existing founder assessment pipeline and web/API into the two approved repositories.
**Architecture:** Keep vc_clone_graph import compatibility in the pipeline. Rename the API package to vclogic_web and use absolute imports from the pipeline. Set an explicit pipeline workspace for API data/config access.
**Tech Stack:** Python, LangGraph, Pydantic, FastAPI, React, Vite, pytest, Vitest.

- [x] Copy source, configs, investor inputs, evaluation references, scripts, and engine tests; exclude generated files and web ownership.
- [x] Copy web/frontend, tests/web, portrait tooling, and API source to the application checkout; rewrite vc_clone_graph.web imports to vclogic_web.
- [x] Update both pyproject.toml files and lockfiles; application uses the sibling pipeline package through a uv source override.
- [x] Add a regression test for separate pipeline workspace and frontend paths; run it before adding workspace support in vclogic_web/app.py and cli.py.
- [x] Document installation, configuration, data ownership, and offline smoke commands in each README.md.
- [x] Run uv run pytest -q in both repositories, deterministic index/run/verify in the pipeline, npm test -- --run and npm run build in web/frontend, and uv build in both repositories. Diagnose extraction regressions against original source.
- [x] Review tracked files and exclusions, record verification results, and commit both extractions locally.
