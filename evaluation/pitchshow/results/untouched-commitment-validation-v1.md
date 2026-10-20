# Untouched Commitment Validation v1

Four previously unattempted VC/pitch pairs, selected before seeing outputs: one In and one Out for Mac Conwell and Michael Hyatt. Founders-only inputs, OpenRouter, target excluded from history. No current or cataloged legacy pair was repeated. No inference code changed while these four workers were running.

## Results

| VC | Pitch | Actual | Predicted | Phase 2 |
|---|---|---|---|---|
| Mac | FranShares #130 | In | Out | accepted |
| Mac | RobeCurls #134 | Out | In | provisional, iteration limit |
| Michael | Cushion #51 | In | Out | accepted |
| Michael | SVRF #45 | Out | Out | provisional, exhausted searches |

**1/4 correct; all four executed successfully; cost $0.46488459.** Both actual Ins were missed. This tiny sample does not establish population accuracy, but provides no evidence that the changes solved the prediction problem.

## What the new fields reveal

- **Mac / FranShares:** The model computes potential fee revenue and treats a $19m post-money price as a concern relative to historical sub-$10m preferences. The live reference instead records Mac saying the price is below $25m and still in range, alongside positive experience with alternative investments. The target's stated price tolerance is correctly held out; the error indicates that the independently retrieved profile/comparisons do not reconstruct it reliably.
- **Mac / RobeCurls:** The model favors realized revenue, profitability and distribution, using RedDrop as a positive consumer-business comparison. It declares that an exception is defeated but chooses conventional fit, which is permitted: that declaration alone is not grounds to reject a conventional In. The actual reference cites potential portfolio-roadmap overlap. The model's main case misses that controlling reason. This is not resolved merely by making the exception schema stricter.
- **Michael / Cushion:** The model values product/adoption but demands stronger transaction economics and expansion proof. The investor actually committed with open questions about the market and lead investor. The independent memory is still being used to impose a stronger commitment bar than the live decision.
- **Michael / SVRF:** Out matches, but the model emphasizes unknown monetization and growth economics; the explicit observed decision reference emphasizes inability to gauge the unfamiliar opportunity. A correct label is not full rationale fidelity.

Reference comments above use the separately stored live-founder-v2 scoring overrides. They were never provided as assessment input. References remain machine-derived and Codex reviewed, not human ground truth.

## Final local hardening after this batch

An additional deterministic test exposed two retry/finalization edge cases: a missing JSON import in the new findings-feedback path, and acceptance of a retained earlier decision based on a newer discarded candidate's sufficiency. Both were fixed after the paid batch finished. A retained reversal now stays provisional. Final targeted validation: **162 tests passed**.

None of the four paid cases recorded commitment-validation findings, missing-exception findings, or the retained-reversal event; these fixes did not address their observed prediction errors. The paid batch therefore validates the main new behavior before those final rare-path fixes; those fixes are verified by tests, not another paid repetition. No second paid run was made.

## Interpretation and preservation

The implementation now makes commitment support, soft-negative weighing and same-risk judgments explicit and checks their bindings. It does not independently verify the truth of a model's semantic judgments. These results indicate that source/profile applicability, omitted investor-specific constraints and calibration of decision weight remain unresolved. Do not treat more detailed explanations as demonstrated predictive improvement.

Previous results and references were preserved. This validation cohort is separately cataloged and included in overall reporting without being relabeled as a development regression. Known wiki source-lineage and aggregate leakage gaps remain; no leakage-clean accuracy claim is made.

See [machine-readable progress and native result paths](untouched-commitment-validation-v1-progress.json), [implementation documentation](../../../docs/COMMITMENT_CALIBRATION.md), and [development reference review](repaired-v3-live-reference-review.json).
