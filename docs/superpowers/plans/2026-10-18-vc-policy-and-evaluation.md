# VC Policy and Evaluation Improvements

> For agentic workers: execute the bounded assignments below with focused regression tests. User approved the design and requested Codex Astra implementation. Preserve unrelated workspace edits and historical artifacts; no commits or paid assessment batch required.

**Goal:** Apply shared policy safeguards, context preservation and automated assessment evaluation to all installed VCs.

**Architecture:** Add three independently switchable default-true booleans under `[assessment]`: `policy_summary`, `fact_policy_comparison`, `exception_evidence`. Use only sanitized, target-excluded wiki evidence, preserving source references, qualification and dates. Integrate deterministic policy selection and explicit decision checks with the existing v4.1 workflow rather than creating a separate evaluator. Preserve historical A/B input bytes and create an explicitly separate audited context-restored variant. Evaluate completed runs automatically against candidate transcript references and classify issues for review.

**Tech stack:** Python, Pydantic, LangGraph, existing retrieval/artifact tooling and pytest.

## Tasks
- [x] Core pipeline: configuration defaults and independent off paths; deterministic compact policy evidence inclusion; sourced fact-policy comparisons; evidence-backed exception checks; integrate settings, prompts, validation and auditable artifacts. Do not invent hard thresholds or treat missing rules as satisfied.
- [x] Input quality: detect orphan answers after speaker filtering; produce auditable separately named context-restored inputs from safe pre-decision source only. Preserve strict founders-only and existing A/B semantics. Never derive missing units from decision explanations.
- [x] Evaluation: compute exact-label and label-direction metrics after each completed assessment, preserve candidate-reference qualification, classify failures, and flag reference issues. Summarize by VC and input mode without mixing backend/model or historical runs.
- [x] All-VC verification: validate every installed canonical config defaults enabled; prepare/audit all eligible cases; evaluate existing results offline; run focused regression and leakage tests. Preserve old predictions and references.
- [x] Documentation: show booleans, disable examples, audit/report locations, context-restored mode, and limits. Record verification and remaining limitations.

## Validation
Test each switch independently, default loading for all 19 VCs, quoted policy provenance and target removal, absent/conflicting/time-ambiguous policy handling, source-backed exceptions, no decision-derived context, A/B byte preservation, strict metrics versus neutral questions, failed assessments not scored, and historical output preservation. Use deterministic tests/offline corpora; no claims of improved prediction accuracy without a future held-out assessment comparison.

## Verification Results

- All 19 canonical investor configs inherit all three enabled settings and retain OpenRouter.
- All 19 wiki indexes supply bounded selected policy evidence (2,663–7,021 characters); Jenny includes qualified numerical valuation guidance. Selection is heuristic, not a guarantee of exhaustive policy coverage.
- Audited 404 VC–episode pairs; preserved all 808 original A/B pitch files and added 404 opt-in restored inputs with 548 neutral topic labels.
- Offline report covers 336 execution records (331 completed, 5 failed), with coverage entries for all 19 VCs. Only eight VCs currently have saved runs. Historical cohorts and providers remain separate.
- All 1,546 protected original pitch, rationale-reference and native summary files retain their pre-change hashes.
- Runtime tests cover independent policy evidence, comparison and exception switches. No paid assessment rerun or accuracy improvement is claimed.

### Limits
Citation checks enforce structural coverage; semantic policy comparison remains model-driven. Context audits flag heuristic candidates, including false positives, and restore only safe adjacent question topics. Automated rationale references remain candidates. Three rehearsal bridge tests require unavailable archived fixtures.
