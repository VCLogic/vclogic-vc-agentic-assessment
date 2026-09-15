# Repository extraction — 2026-09-15

Source: working files in `/home/dpasch01/vc-digital-twins/langgraph-vc-clone-framework`, based on commit `6cad8c570892d9362d1f2007fec9116a5566ee98` plus local changes. The source checkout was not modified.

## Ownership

This repository owns the `vc_clone_graph` engine, assessment/rehearsal CLI, configurations, investor inputs, evaluation references, and engine tests. The separate [web repository](https://github.com/VCLogic/vclogic-web-application) owns the `vclogic_web` API and React frontend. It installs this distribution through an editable sibling source for local development or an engine wheel for deployment.

No environment, secret file, generated index, checkpoint, runtime output directory, or frontend code was imported here. The small frozen v4.3 canary manifest under `reports/evaluation` is retained as a benchmark input. Historical generated assessment runs are external prerequisites for archive-dependent evaluation commands. Existing evidence bytes are preserved, including their original whitespace, to retain manifest hashes.

## Compatibility changes

- The distribution is named `vclogic-vc-agentic-assessment`; Python imports and engine CLI names remain unchanged.
- FastAPI, Uvicorn, and the web CLI belong to the web distribution.
- Rehearsal configurations accept an explicit runtime workspace. Canonical run configs derived by bootstrap carry that workspace through input, index, output, and checkpoint resolution. The runtime workspace is not serialized into configuration hashes. Standalone CLI behavior remains unchanged when no workspace is supplied.
- The personalized evaluation taxonomy now uses this repository's `inputs/taxonomy/codebook_v_final.json`.

## Verification

Full engine run: 1,680 passed, 16 failed, 3 optional-dependency skips. All failures were missing historical artifacts. After retaining the benchmark manifest and making archived-output prerequisites explicit, all 16 affected checks were rerun: 1 passed, 15 archival checks skipped. Combined result: **1,681 passed, 18 archival/optional skips, no unresolved failures**. The expensive numerical model tests were included in the full run.

Additional checks:

- Workspace/configuration/CLI/bootstrap regression suite: 121 passed.
- Reference/configuration migration suite: 114 passed, 1 archival skip.
- Deterministic fake-provider index, both assessment phases, and artifact verification: passed.
- Source distribution and wheel builds: passed.
- Isolated wheel installation, including the web wheel: passed API health and investor listing checks.
- Standalone engine wheel executed and verified both deterministic assessment phases from another working directory.
- Reviewer confirmed import ownership and workspace propagation. A discovered canonical-runner cwd regression was reproduced, corrected, and rechecked.

The web repository separately records 83 API tests, 89 frontend tests, and 6 desktop/mobile browser smoke checks passing. No paid assessment was run.
