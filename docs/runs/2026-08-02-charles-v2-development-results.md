# Charles rationale contract v2: development results

## Scope

This report evaluates the versioned v2 rationale contract on four deliberately selected Charles Hudson error-analysis cases: Dig, Pepper, Harper Wilde, and Nectir. These cases helped motivate the redesign, so the results are diagnostic development evidence—not a holdout estimate and not a publication-grade performance claim.

The corrected pitch-window labels are three Ins (Dig, Pepper, and Nectir) and one Out (Harper Wilde). A stated willingness to invest subject only to a routine conflict check counts as In; this makes Nectir an In. Every inference input remained pitch-only. Charles's questions, other panelists' comments, and post-pitch explanations were used only after inference for the reasoning audit.

## Final results

| Episode | Actual | Any-check | Likelihood | Confidence | Standard check | Review score | P1 / P2 status |
|---|---:|---:|---:|---:|---:|---:|---|
| Dig | In | **In** | 0.62 | 0.64 | Out (0.35) | 0.68 | accepted / accepted |
| Pepper | In | **Out** | 0.30 | 0.82 | Out (0.18) | 0.62 | accepted / provisional |
| Harper Wilde | Out | **In** | 0.53 | 0.68 | Out (0.16) | 0.75 | accepted / provisional |
| Nectir | In | **In** | 0.60 | 0.68 | Out (0.30) | 0.74 | accepted / provisional |

At the fixed 0.50 any-check threshold, v2 is correct on 2/4 cases: 2 true Ins, 1 missed In, 1 false In, and 0 true Outs. In precision, recall, and F1 are each 66.7%; accuracy is 50.0%. Balanced accuracy is only 33.3% because the sole Out is missed. The selected set is too small and class-skewed for these percentages to be stable.

The binary predictions are the same as the historical Codex-orchestrated results—Dig, Nectir, and Harper In; Pepper Out—although the reasoning contract is materially different. Relative to the immediately preceding adaptive LangGraph/Luna run, v2 repairs Nectir but regresses Harper. The net classification count remains 2/4.

## Diamond-search ranking

For prioritization, the coherent ranking endpoint is any-check likelihood, not the separate model-generated `ranking_score`.

| Rank | Episode | Actual | Any-check likelihood |
|---:|---|---:|---:|
| 1 | Dig | In | 0.62 |
| 2 | Nectir | In | 0.60 |
| 3 | Harper Wilde | Out | 0.53 |
| 4 | Pepper | In | 0.30 |

Any-check likelihood gives average precision 0.917 against 0.75 In prevalence and ROC AUC 0.667. The top-two review queue contains two of the three actual Ins with 100% precision; reviewing all four is required to recover Pepper. The separate `ranking_score` performs worse: it ranks Harper first, then Nectir, Dig, and Pepper, yielding AP 0.639. It should remain an audit field, not the diamond-search endpoint, until its semantics are redesigned and calibrated.

## What v2 fixed

### Total round size no longer becomes an automatic veto

Dig's $1.5 million round and Nectir's $3 million round are recorded separately from a possible Charles check. Phase 2 treats the larger rounds as size-limiting rather than fatal. Both companies receive exploratory sub-$100K Ins, which matches Charles's observed small-check logic: $10K for Dig and stated $50K–$100K intent for Nectir.

### Routine portfolio adjacency no longer becomes a fatal conflict

Dig's pet holdings adjacency and Nectir's edtech holdings adjacency are explicitly recorded as routine and non-blocking. Nectir's Phase 2 says the Campus/ClassDojo/Wonder School adjacency does not establish product, customer, channel, or grievance overlap. This repairs the preceding run's central Nectir error.

### Phase 2 escalation is now detectable and auditable

The validator prevents a Phase 2 fatal risk unless Phase 1 recorded affirmative-adverse evidence with fatal severity. Complete decisions that violate provenance are retained as provisional rather than discarded. Pepper and Harper are therefore available for analysis, with the exact negative-direction inconsistency recorded in their summaries.

### The taxonomy now exposes previously hidden dimensions

The v2 investigations explicitly examine capital efficiency, right-sized check optionality, exit-path alignment, and founder/investor vision alignment. This makes the reasoning more inspectable even when the model's conclusion on a dimension is wrong.

## What v2 did not fix

### Pepper remains a false negative

Phase 1 recovers the specific customer pain, custom cup molds, reported $25 CAC and $130 LTV, Urban Outfitters reorder, founder-market connection, coachability, and possible category expansion. It also records True & Co as a material but unresolved portfolio overlap and the $1.5 million round as material fund-model friction. Phase 2 does not call either fatal, but it still treats both as controlling reasons to commit no capital.

This misses Charles's actual small-check exception logic: he liked the founder, saw capital efficiency and a sharply identifiable beachhead, and invested $25K. It also reflects a hard information limit of the pitch-only task: Charles explicitly updated his market-size belief after hearing the other panelists, but those panel comments are correctly excluded from the pitch-only input. V2 prevents an invented fatal conflict, but it does not yet calibrate when a material review item should coexist with an exploratory In.

### Harper becomes a false positive

Phase 1 now contains an explicit `founder_investor_vision_alignment` rationale, but marks the issue unresolved rather than adverse. It recognizes that the founders' mission is clear while the vision beyond bras is not. Phase 2 then allows a 0.53 exploratory In because of customer research, the 400-bra test, capital efficiency, and early behavioral signals.

Charles's actual reason for passing was stronger and more interpersonal: he did not share the founders' view of the brand principles and brand anchor. The pitch supplies the founders' brand thesis but not Charles's reaction to it. The agent therefore treats absent alignment evidence as uncertainty and buys a small option, whereas Charles treated observed misalignment as decisive. This is now the clearest remaining error: the system identifies the right dimension but does not infer Charles's negative judgment from the pitch and memory.

### Provenance enforcement detects synthesis drift but does not repair it

Pepper and Harper are provisional because Phase 2 assigns a negative effective direction to a rationale that Phase 1 marked unresolved rather than affirmative-adverse. Retaining and flagging these outputs is preferable to losing the decisions, but the model still needs a stronger synthesis instruction—or a deterministic direction mapping—to avoid the inconsistency in the first place.

### Risk-ledger completeness remains imperfect

Nectir is provisional because its risk ledger does not exactly match every opposing rationale ID. The decision and its controlling logic are coherent, but the bookkeeping contract is incomplete. This is an output-quality issue rather than a classification error.

## Reasoning fidelity by episode

### Dig

The v2 rationale set is broad and substantially aligned with Charles's observable reasoning: founder authenticity and communication, problem insight, early adoption, acquisition economics, marketplace-balance risk, incumbent competition, monetization uncertainty, and limited venture scale. It adds explicit exit-path and small-check optionality fields. The final exploratory In is faithful to Charles's behavior. The remaining mismatch is that the system still describes round size as fund-model friction, whereas Charles's actual $10K offer demonstrates considerable flexibility.

### Pepper

Extraction is strong, but weighting remains too conservative. The material True & Co review and round-size concern overshadow capital efficiency, founder appeal, early economics, and the focused beachhead. Unlike the prior adaptive run, v2 does not invent a fatal conflict; nevertheless, it still fails to reproduce the exploratory $25K exception. The missing panel-induced market update is an irreducible limitation of pitch-only forecasting for this episode.

### Harper Wilde

V2 correctly retrieves the True & Co precedent as an unresolved review rather than a fatal conflict and explicitly examines vision alignment. However, it interprets vision alignment as missing evidence, not demonstrated mismatch, and the positive execution/capital-efficiency cluster pushes the any-check likelihood just above threshold. The reasoning is more structurally faithful than the historical false positive, but it still misses Charles's decisive negative brand-thesis reaction.

### Nectir

This is the strongest repair. V2 captures paid validation, campus deployment, founder execution, go-to-market motion, sales-cycle uncertainty, recurring pricing, capital efficiency, LMS integration, vision alignment, valuation friction, round/check separation, and routine edtech adjacency. The exploratory In at 0.60 matches Charles's stated $50K–$100K intent subject to a routine conflict check. Entry valuation remains a material adverse consideration but no longer becomes an absolute veto.

## Usage and failed development attempts

| Final run | Input tokens | Cached input | Output tokens | Cost |
|---|---:|---:|---:|---:|
| Dig | 61,650 | 2,140 | 14,220 | $0.01599 |
| Pepper | 76,371 | 0 | 19,193 | $0.02106 |
| Harper Wilde | 39,802 | 0 | 12,430 | $0.01241 |
| Nectir | 127,242 | 3,107 | 19,037 | $0.02697 |
| **Successful final artifacts** | **305,065** | **5,247** | **64,880** | **$0.07644** |

During implementation, four failed attempts exposed and motivated three contract corrections: retaining provenance-flagged decisions, allowing evidenced routine adverse concerns, and allowing a named dimension to be explicitly cleared with severity `none`. Those failed attempts consumed another 430,246 input tokens, 65,594 output tokens, and $0.09282. Total live development usage was 735,311 input tokens, 130,474 output tokens, and $0.16926. The failed-attempt cost is engineering/debugging overhead and should not be used as the expected cost of a frozen production run.

Nectir's final input volume is unusually high because two Phase 1 passes and two Phase 2 passes repeatedly carry large exact wiki evidence and frozen rationale payloads. Before scaling or switching to a more expensive model, the next cost intervention should reduce exact-read payload size and avoid resending unchanged evidence while retaining chunk-level provenance.

## Conclusion

V2 successfully repairs the two specific synthesis pathologies it was designed to constrain: total round size no longer implies that Charles cannot make a small check, and routine category adjacency no longer becomes a fatal conflict. That produces a reasoning-faithful Nectir In and retains the correct Dig In. It does not improve aggregate classification on these four development cases because Pepper remains too conservatively weighted and Harper's true brand-vision mismatch is not recoverable from the pitch-plus-memory inference as currently implemented.

The main remaining scientific problem is therefore narrower than before: not rationale discovery in general, but investor-specific threshold and exception calibration—especially when positive founder execution justifies a small option, and when an apparently attractive, capital-efficient test should still be rejected because the investor does not share the founder's core vision. The contract should be stabilized on development data and then evaluated on untouched episodes; these four cases must not be presented as holdout evidence.
