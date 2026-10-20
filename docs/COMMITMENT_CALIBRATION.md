# Commitment Calibration and Live-Pitch Evaluation

These changes implement the [21-case error-analysis recommendations](../evaluation/pitchshow/results/repaired-v3-error-analysis.md). They apply to future v4.1 assessments for every canonical VC through the shared workflow. Prior predictions remain unchanged.

## Assessment changes

With `fact_policy_comparison = true` (the existing default), an In must supply `commitment_support`: positive pitch-grounded rationale IDs, accessible investor evidence, an affirmative case, the strongest soft negative, and why the positive evidence outweighs it. Lack of a hard veto is explicitly insufficient. A soft preference may justify an Out without becoming an absolute rule.

With exception checking enabled, the model must assess whether the precedent supports the same controlling risk and whether a material difference defeats the analogy. An unsupported or defeated assessment prevents an accepted In; the model can revise on later iterations. At exhaustion the candidate remains provisional with explicit findings, rather than silently changing its decision. Existing evidence/format checks also block acceptance.

This is a **structured self-assessment**, not an independent semantic judge. Citation binding and declared contradictions are checked in code; a model that incorrectly declares an analogy valid can still pass. Untouched-case validation and direct inspection remain necessary. There is no claim that these checks guarantee accuracy.

The instructions require investor-specific weighing of category preference, valuation, competition, coachability and communication. Transaction economics must distinguish GMV from revenue and consider obtainable merchants, adoption, net take and costs. Missing figures remain unknown; the system must compare the investor's demonstrated tolerance of uncertainty rather than demand mature-business metrics universally.

Legacy decision JSON remains readable through a default unassessed support object. Old records are never retroactively converted into accepted new-model decisions.

## Reference protocol correction

The former extraction prompt intentionally included the immediate investor-only debrief as original pitch. The new protocol counts only discussion while the founder remains present, including live decisions, negotiation and reversals. Post-exit discussion is retained as later commentary. This is a protocol change, not evidence that the old extractor violated its prompt.

`python scripts/review_live_pitch_references.py` publishes source-hashed overrides under `evaluation/pitchshow/rationale-corpus/live-founder-v2/`. Original reference files are untouched. Individual mentions are reaggregated after timing corrections, so a later negative remark cannot replace an earlier live assessment of the same label. Rerunning with different content refuses to overwrite an existing override.

Coverage at implementation: 82 VC/pitch references across 23 reviewed episodes; 266 post-exit mentions moved. This covers all 21 development assessments and the four fresh validation cases. Other existing episodes retain their original reference scope until reviewed. Future extractions use the corrected instructions and version fingerprint. No automatic paid re-extraction of the full corpus is triggered.

`evaluation/pitchshow/results/repaired-v3-live-reference-review.json` records source-verified decisive quotes and Codex label/direction review for all 21 development cases, with explicit ambiguity flags. Reviews are Codex inspections and machine references, not human validation; original taxonomy labels remain candidate labels.

## Evaluation changes

The evaluator prefers live-founder overrides, records reference scope, and separates scopes in aggregate groups. It reports:

- Missing decisive labels.
- Labels detected with a different direction.
- Correctly directed labels not included among controlling rationale IDs.
- Correctly directed controlling labels where the final decision still disagrees.

These are exact-label diagnostics, not proof that semantic synonyms are wrong or that a particular weight caused the error. Old evaluation reports remain immutable; recalculation creates a new report.

## Fresh validation

`untouched-commitment-validation-v1` uses four previously unattempted cases: one In and one Out for Mac and Michael, the only investors with both outcomes remaining after current and legacy exclusions. Selection uses observed labels only for class balance, then lowest episode number; predicted outputs are not used. It is separately cataloged as validation and is not pooled with the development cases. The sample is too small to establish population accuracy.

Known wiki lineage and aggregate leakage gaps remain. Future sources remain allowed if they do not disclose the current target's decision or rationale. No leakage-clean performance claim is warranted.

## Validation outcome

[Fresh four-case results](../evaluation/pitchshow/results/untouched-commitment-validation-v1.md): 1/4 decision matches, no execution failures, $0.4649. This does not demonstrate improved accuracy. Final code passed 162 targeted tests, including retained-decision acceptance and findings-feedback paths hardened after the paid batch. None of the four cases exercised those final fixes.
