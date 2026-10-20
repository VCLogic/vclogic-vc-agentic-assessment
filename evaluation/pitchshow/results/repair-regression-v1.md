# Assessment Repair Regression

Implemented the approved repair plan. Validation: **338 targeted tests passed**. Published 495 historical records across 19 canonical VCs (85 recovered, 180 still excluded), rebuilt local indexes, refreshed 246 manifests, and verified 5,280 prior native result files unchanged.

## Four-Case OpenRouter Comparison

These are deliberately selected diagnostic reruns of known failures, not independent benchmark cases. They use new execution identities and the separate `repair-regression-v1` cohort. Original results and leakage labels remain in the overall catalog.

| Case | Actual | Before | Repaired | Phase 2 |
|---|---|---|---|---|
| beck-bamberger-bad-ideas-group / 128-churchspace-airbnb-for-churches | In | Out | In | provisional (corpus_search_exhausted) |
| beck-bamberger-bad-ideas-group / 121-ride-iq-peloton-for-horses | Out | In | In | provisional (iteration_limit) |
| jake-chapman-marque-ventures / 8-hykso | In | Out | In | provisional (corpus_search_exhausted) |
| elizabeth-yin-hustle-fund / 186-supercool-earth-making-it-rain-literally | Out | In | Out | provisional (corpus_search_exhausted) |

**Decision matches: 0/4 before, 3/4 after.** RideIQ remains wrong. All four decisions remain provisional, so this does not establish converged predictions or a reliable overall accuracy improvement.

Observed reasoning changes:

- ChurchSpace now has 9 accessible precedents, including 2 Ins; it uses the restored KredFeed commitment rather than treating a single Out as the whole history. BAM's PR-client profile is absent from the policy bundle. The generated In uses conventional fit and an exploratory commitment.
- Hykso explicitly treats the later defense mandate as insufficient proof of a target-era veto, and compares product/demand evidence with historical hardware decisions.
- Supercool remains Out after the comparison; the justification tests unpaid pilots, unpriced LOIs, sales assumptions, operating costs, and the limits of the In analogies.
- RideIQ still places too much weight on the proposed expansion path relative to the actual investor's concern about going beyond horses. The repair does not resolve every judgment error.

## Output-Format Repair

The four-case run still encountered malformed plans. A separate replay of a malformed ChurchSpace planning request through JSON-object compatibility mode produced a valid plan in 562 output tokens. The final adapter now remembers malformed structured output per phase and uses compatibility mode on the **next scheduled call**; it does not add hidden paid retries. Unit tests cover that transition and preserve raw responses and usage.

That final adapter change was applied **after** the four-case comparison to keep its code stable during execution. Its actual compatibility mode was exercised by the separate probe; the four complete assessments were not rerun again after this adapter change. A final client-profile filter refinement also preserves genuine investment policies about agencies; the previous filter could overexclude them.

Four-case cost: **$0.5063**. Separate compatibility probe: **$0.0022**. Total: **$0.5085**.

## Remaining Limits

- Wiki `full:` source-lineage exclusion and target-inclusive aggregate leaks are still tracked separately. This regression is not certified leakage-clean.
- Applicability and exception checks enforce explicit evidence and comparison structure; they cannot prove semantic correctness.
- All four final Phase 2 decisions remain provisional. Search exhaustion is reported explicitly rather than silently relabeled as convergence.
- No new all-VC batch was launched after the repair.

[Live/final regression data](repair-regression-v1-progress.json) · [Frozen selection](repair-regression-v1.json) · [Compatibility probe](repair-plan-fallback-probe.json) · [Overall results](overall-campaign-results.json) · [Implementation notes](../../../docs/ASSESSMENT_REPAIR.md)
