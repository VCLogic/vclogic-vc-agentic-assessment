# Leakage Audit of the 36-Case Experiment

**Finding: the batch contains confirmed indirect leakage. The reported 24/36 is a raw, contaminated result and must not be presented as leakage-clean predictive accuracy.**

Audited all 36 saved runs: 472 assessment prompts, 32 post-decision reflection prompts, 672 historical-retrieval artifacts, and 1,386 retained pitch lines. No paid inference was run. Original prediction and prompt artifacts retain their hashes.

## Confirmed Cases

| Target VC / pitch | Exposure in actual prompts |
|---|---|
| Nicole Verkindt / ShearShare | Wiki disclosed a $25,000 SAFE at a $6m cap under “Could This Be the Next Airbnb?” and an unnamed-company paraphrase. Citation `ctx:nicole-verkindt-0234` resolves the target. Evidence `W-910e677581920e697e91` and `W-b4dc4cc2001c2f44e988` appeared in the first investigation and are cited in the final decision. |
| Jake Chapman / Babyscripts | Shimmur's intro repeats Jake's held-out investment rationale about positive impact and making a “boatload of money.” `H-eafc128637293eca24ab`. |
| Howie Diamond / Babyscripts | The same intro exposes Jake's held-out comments about the target pitch, although Jake is another panelist in this evaluation. `H-5111e3082088ac1ea3fe`. |
| Jake Chapman / Industrial Organic | Shimmur's intro repeats Jillian's target-pitch conviction statement, “There is no doubt in my mind that you're going to do this.” `H-eafc128637293eca24ab`. |
| Phil Nadel / iMirror | Sudden Coffee's intro repeats Howie's target-pitch objection about accomplishing the same thing on an iPad. `H-ecc2c9dbfeaf9b612e26`. |

These five predictions matched their reference decisions. Exposure is proven; its causal effect on those decisions is not. Some exposures concern another panelist's assessment rather than the target VC's own decision, but they still introduce held-out pitch commentary into a founders-only evaluation.

## Additional Suspicious Cases

- **Phil / Hykso:** a short phrase about members not buying and the business falling apart occurs both in the target rationale and in Skylights' intro. Exposure is verified, but the origin of this short phrase is not conclusively established. The synthesis classifies this more cautiously than the historical component report.
- **Paige / Lotus:** exposed show-portfolio totals include the target investment.
- **Rohit / Mappa:** exposed show-portfolio totals include the target investment.

The remaining **28 cases have no evidence found within this audit**, not a guarantee against every paraphrase. The founders-only pitch fields themselves showed no confirmed decision/rationale leakage. Evaluation reference labels were not observed being fed into inference.

## Why Earlier Checks Missed These

Company aliases do not catch episode-title or unnamed-company disclosures. Another episode can include a montage copied from the held-out pitch without its name. Wiki citation exclusion also fails to follow `full:` source IDs: an adversarial test hid a target source entry while preserving a decision paraphrase and its derived summary. Aggregate statistics can retain target contributions after individual rows are removed.

Later sources are permitted under this experiment's rules. These findings concern target-derived content, not publication dates.

## Tracking and Next Steps

All 36 execution jobs have a separate `leakage-audit.json`; the progress ledger links to this audit. Original results are preserved and no reruns were charged. Confirmed and suspicious cases are quarantined from clean-accuracy claims. Do not infer a corrected clean accuracy by simply dropping these cases.

Before further validation, exclude target source identities and their derived claims, strip/deduplicate promotional intros and recaps across episodes, and remove or recompute target-inclusive aggregates. Then audit the exact model inputs and rerun affected cases under new execution identities. Source-based review cannot establish whether a model already knows an episode from training.

## Evidence

- [Combined classifications](leakage-audit-36-summary.json)
- [Wiki provenance and exact prompt paths](leakage-audit-36-wiki.json)
- [Historical-source excerpts and target turns](leakage-audit-36-precedents.json)
- [Pitch-to-source mappings](leakage-audit-36-inputs.json)
- [Adversarial source-link test](leakage-audit-adversarial-probes.json)
- [Preserved artifact hashes](leakage-audit-36-artifacts.json)
