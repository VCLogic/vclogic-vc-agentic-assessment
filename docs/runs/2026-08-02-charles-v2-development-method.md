# Charles rationale contract v2: development method

## Purpose

This run tests whether a stricter, evidence-linked rationale contract repairs reasoning errors diagnosed in four Charles Hudson cases: Dig, Pepper, Harper Wilde, and Nectir. These episodes were selected because their prior outputs informed the redesign. They are development and error-analysis cases, not an untouched holdout and not an unbiased performance estimate.

## What remains fixed

- The pitch-only episode inputs, Charles investment-memory wiki, source-linked wiki evidence, and package manifests remain unchanged.
- Prior v1 configurations and output artifacts remain immutable and reproducible.
- Generation uses `openai/gpt-5.6-luna` through OpenRouter; retrieval embeddings use local Ollama `nomic-embed-text`.
- Phase 1 permits at most three plan/retrieval/investigation passes. Phase 2 permits at most two synthesis passes.

## What changes

- The run selects `taxonomy/codebook_v2.json`, which retains the original 44 rationales and adds capital efficiency, right-sized check optionality, and exit-path alignment.
- Phase 1 emits `investigation-v2`. Each rationale records its evidence status, constraint kind, constraint severity, and severity basis. Financing facts are recorded separately as total round size, stage, entry valuation, possible Charles check, lead requirement, and ownership feasibility.
- Phase 2 emits `decision-v2` and is validated against the complete frozen Phase 1 artifact. It cannot upgrade unresolved or routine evidence into an affirmative adverse or fatal constraint. Total round size cannot become a fatal constraint by itself.
- Unknown information is retained as uncertainty and may reduce likelihood, confidence, or check size; it is not silently converted into negative evidence.

## Evaluation

Each case is regenerated from the beginning, then compared with its v1 rationale set, v1 decision, corrected observed label, and Charles's actual episode language. The audit reports classification, any-check likelihood, ranking order, rationale fidelity, convergence status, token usage, and provider-reported cost. Because the cases shaped the intervention, results can diagnose whether the proposed mechanism behaves as intended but cannot establish generalization.
