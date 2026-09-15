# Evaluation labels

This directory is outside `inputs/` and must never be exposed to Phase 1 or
Phase 2 inference. It contains outcomes used only after a run has frozen its
predictions.

`charles_pitch_window_decisions.json` is a byte-identical copy of:

`../agentic-vc-clone-framework/data/annotations/charles_pitch_window_decisions.json`

Copied on 2026-08-01 with SHA-256:

`4f8baabe560ee7a9268a4c42eedc1a81c028f810488ef813720efa63bd4bd85f`

The file contains 113 audited Charles episode records. Of these, 86 are marked
evaluation-eligible: 15 final `In` and 71 final `Out` decisions. Evaluation must
use `final_decision` as the corrected outcome label. Fields such as
`pitch_window_decision`, `initial_response`, and `decision_context` are retained
for error analysis and must not be substituted for the corrected label.
