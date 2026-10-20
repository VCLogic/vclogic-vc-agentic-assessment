# Assessment Repair

The fixes follow the [first-11 diagnosis](../evaluation/pitchshow/results/round2-first11-diagnosis.md). They apply to new executions; saved results remain unchanged.

- Historical preparation supports both independently verified receipt formats. A verdict must match the target VC and exact evidence; explicit rejections cannot be overridden. Use `scripts/prepare_pitchshow_history.py --source-backup <original-install-backup> --output-root <new-stage>` to rebuild from originals without replacing installed data first.
- Policy summaries exclude explicit PR/consulting client-profile descriptions and include role/era qualifications. A hard veto needs an `established` applicability assessment with a bound source citation. Unknown applicability prevents an accepted decision. Later publications remain allowed; they must describe the relevant role/era rather than automatically impose today's fund mandate.
- Founder exceptions require a comparison stating the controlling concern, cited observed In, similarities and differences. The model must address venture-scale economics using supplied quantities and distinguish interest from a concrete commitment. These structural requirements do not certify semantic analogy quality.
- Retrieval questions/queries are bounded to 300 characters; continuation focus to 600. Canonical planning budgets are 4,096 output tokens. OpenRouter response metadata records finish reason, token-limit exhaustion and malformed/repetitive formatting. Raw responses and usage remain preserved for audit; no model output is silently rewritten.
- Exhausted searches and iteration limits receive distinct stop reasons. Provisional results remain provisional; the repair does not relabel them as accepted merely to improve reported completion.

The existing `policy_summary`, `fact_policy_comparison`, and `exception_evidence` flags remain independently configurable and default enabled. The known wiki source-lineage and target-inclusive aggregate leakage gaps remain unresolved. No leakage-clean accuracy claim follows from these repairs.

The four-case regression is diagnostic: it reuses known cases under new execution identities and must not be pooled as new independent benchmark evidence.

Validated with 338 targeted tests and a [four-case comparison](../evaluation/pitchshow/results/repair-regression-v1.md): 3/4 decision matches versus 0/4 previously, all still provisional. Final OpenRouter compatibility fallback is tested separately after the comparison; it adds no hidden retries.
