# Charles: context-rich single-call baseline versus the agentic loop

## Executive conclusion

The single-call baseline and the two-phase agentic loop have essentially the same pitch-classification performance on the matched cohort. The agentic loop recovers slightly more Ins, but does so by predicting even more Outs as Ins. The single-call baseline produces materially better early ranking and costs roughly one-sixteenth as much.

This is important negative evidence for the current agentic design. Iterative retrieval and separate rationale/decision phases have not yet demonstrated incremental predictive value over one large, carefully bounded prompt containing the same investment memory, taxonomy, pitch, and selected precedents.

The agentic system may still be more useful as an auditable interaction process: it records searches, source reads, intermediate questions, and evidence provenance. That process value should be evaluated separately from predictive performance.

## Experiment population

The experiment followed the established deterministic round-robin ordering and stopped immediately after the final observed In. It contains **37 Charles episodes: 19 Ins and 18 Outs**. No later Outs were run.

This is an intentionally balanced development cohort, not the natural base-rate distribution and not an untouched holdout set. Precision and average precision therefore should not be generalized to normal inbound deal flow without base-rate adjustment.

## Compared systems

### Context-rich single-call baseline

One `openai/gpt-5.6-luna` call received:

- the audited pitch-only transcript;
- the complete Charles investment wiki and underlying evidence;
- the complete rationale taxonomy;
- the five most semantically relevant target-excluded precedent episodes, including complete transcripts and audited Charles decisions; and
- one strict schema combining rationale identification with Phase 2-equivalent classification and ranking outputs.

There was no tool loop, separate Phase 1, model repair, or episode retry.

### Two-phase agentic loop

The same Luna model operated through LangGraph. Phase 1 iteratively planned searches, read the wiki and historical precedents, and constructed a frozen rationale record. Phase 2 separately revisited evidence and precedents to synthesize any-check, standard-check, likelihood, confidence, and ranking outputs. Each phase used two iterations, and failed episodes could receive a second episode attempt.

## Output reliability

| Measure | Single-call baseline | Agentic loop |
|---|---:|---:|
| Scheduled episodes | 37 | 37 |
| Decision-usable outputs | 35 (94.6%) | 34 (91.9%) |
| No usable decision | 2 (5.4%) | 3 (8.1%) |
| Fully accepted by the original runtime contract | 11 (29.7%) | 0 fully converged; 34 provisional |

The baseline's low original strict-acceptance figure requires explanation. Twenty-three outputs were complete, schema-shaped decisions but the post-validator expected a bare precedent slug while Luna returned unambiguous forms such as `slug: explanation` or `Precedent 83, observed decision In`. Every one mapped uniquely to one of the five supplied precedents. These are validator-compatibility findings, not missing decisions, and were recovered deterministically without another model call. Two responses—episodes 41 and 62—hit the 16,384-output-token ceiling and were truncated. Episode 170 contained a complete decision but referenced nonexistent rationale `R38` in one nested field; its decision and ranking endpoints remain usable, but its full rationale contract is not valid.

All 34 agentic decisions in this cohort were marked provisional in both phases rather than fully converged.

## Classification

### Each system's available decisions

| System | Cases | TP | FP | TN | FN | Balanced accuracy | In precision | In recall | In F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Single-call baseline | 35 | 14 | 11 | 6 | 4 | 56.5% | 56.0% | 77.8% | 65.1% |
| Agentic loop | 34 | 15 | 13 | 4 | 2 | 55.9% | 53.6% | 88.2% | 66.7% |

### Strict matched comparison

Thirty-two episodes have decisions from both systems: 16 Ins and 16 Outs.

| System | TP | FP | TN | FN | Balanced accuracy | In precision | In recall | In F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Single-call baseline | 13 | 11 | 5 | 3 | 56.25% | 54.2% | 81.25% | 65.0% |
| Agentic loop | 14 | 12 | 4 | 2 | 56.25% | 53.8% | 87.5% | 66.7% |

The systems therefore have the **same matched balanced accuracy**. The agentic loop finds one more In but also generates one more false In. Neither is a reliable final classifier at the present threshold. Both behave as high-recall screening systems.

The systems agree on only **62.5%** of matched binary decisions. This is not evidence that they are learning the same policy through different implementations.

## Prioritization or diamond search

The explicit `ranking_score` is the strongest single-call endpoint.

### Each system's available decisions

| System | AP | ROC AUC | Ins in top 5 | Ins in top 10 | Ins in top 20 |
|---|---:|---:|---:|---:|---:|
| Single-call baseline | **0.759** | **0.734** | 4/5 | 9/10 | 13/20 |
| Agentic loop | 0.633 | 0.663 | 3/5 | 8/10 | 12/20 |

The baseline prevalence is 18/35 (0.514), while agentic prevalence is 17/34 (0.500). Both exceed prevalence, but the single-call baseline has a clearer early-ranking advantage.

### Strict matched comparison

| System | AP | ROC AUC | Ins in top 5 | Ins in top 10 | Ins in top 20 |
|---|---:|---:|---:|---:|---:|
| Single-call baseline | **0.741** | **0.707** | 4/5 | 9/10 | 11/20 |
| Agentic loop | 0.632 | 0.660 | 3/5 | 8/10 | 12/20 |

The baseline is better when the review budget is scarce: four of its top five and nine of its top ten are actual Ins. The agentic loop catches one more In by top 20. Their rankings are only moderately related (Spearman rho = 0.489, p = 0.0045), so the architecture materially changes which opportunities receive attention.

## Rationales

Across the 32 episodes with both a baseline rationale record and an agentic Phase 1 record:

| Comparison | Result |
|---|---:|
| Mean rationales per baseline pitch | 29.2 |
| Mean rationales per agentic Phase 1 pitch | 25.4 |
| Mean rationale-label Jaccard similarity | 63.0% |
| Agentic labels also found by the baseline | 83.7% |
| Direction agreement on shared labels | 71.4% |

The agentic loop does not discover a broader rationale set. The one-call baseline identifies about four more rationales per pitch and recovers roughly 84% of the labels produced by the agentic process. The moderate Jaccard and direction agreement show that the two procedures still interpret meaningful portions of the same pitch differently. The next scientific comparison should use Charles-derived ground-truth rationales, not treat either model-generated rationale set as truth.

## Economics

| Resource over the 37 scheduled episodes | Single-call baseline | Agentic loop | Agentic/baseline |
|---|---:|---:|---:|
| OpenRouter cost | **$0.745** | **$12.001** | 16.1x |
| Input tokens | 3,518,001 | 59,123,580 | 16.8x |
| Output tokens | 509,379 | 2,030,096 | 4.0x |
| Mean realized cost per scheduled pitch | $0.020 | $0.324 | 16.1x |

The agentic loop's cost includes realized second attempts where applicable. This is the appropriate operational comparison: it represents what was actually required to obtain—or fail to obtain—the recorded outputs.

## Where the systems differ

On the 32 matched cases, the baseline alone was correct for episodes 18, 68, 43, 65, 66, and 175. The agentic loop alone was correct for episodes 77, 104, 61, 63, 149, and 69. Both were wrong for episodes 20, 22, 24, 37, 38, 52, 54, and 71.

The agentic loop's any-check mechanism is more permissive. It improves In recall but repeatedly converts uncertainty and optionality into speculative small-check Ins, including false positives such as RowVigor. The baseline is also permissive, but its explicit ranking field separates attractive review candidates more effectively than its binary endpoint.

## Interpretation

1. **The current agentic loop has not earned its additional complexity on prediction.** Matched balanced accuracy is identical, ranking is worse, and cost is sixteen times higher.
2. **The single-call baseline is currently the stronger diamond-search method.** Its ranking score is the best endpoint in this experiment, particularly at top-five and top-ten review budgets.
3. **Neither binary endpoint is deployment-ready.** False positives remain high, and the balanced cohort obscures the much lower precision expected under natural deal-flow prevalence.
4. **The likely value of agency is process, not prediction—so far.** Search traces, explicit source reads, separable rationale formation, and auditable reconsideration may help founders reflect and may support research into decision logic. Those benefits require their own evaluation criteria.
5. **The comparison strengthens the project's scientific framing.** It directly answers “why not just give ChatGPT everything?” At present, doing exactly that is competitive or superior on outcomes. The contribution cannot be claimed as predictive improvement from agentic orchestration. It can instead be tested as an auditable instrument for engaging with investor-specific rationales and source-linked decision processes.

## Episode-level appendix

`FAIL` means no decision-usable endpoint for that system. Baseline likelihood and rank remain available for episode 170 despite its nested rationale-reference defect, so it is included as decision-usable.

| # | Episode | Actual | Baseline | B likelihood | B rank | Agentic | A likelihood | A rank |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | `39-this-pitch-is-damn-near-perfect` | In | Out | 0.20 | 0.58 | FAIL | — | — |
| 2 | `18-rowvigor` | Out | Out | 0.05 | 0.18 | In | 0.36 | 0.31 |
| 3 | `41-can-this-startup-help-retailers-take-on-amazon` | In | FAIL | — | — | In | 0.56 | 0.56 |
| 4 | `20-harper-wilde` | Out | In | 0.56 | 0.63 | In | 0.62 | 0.58 |
| 5 | `68-dogs-dating-a-match-made-in-heaven` | In | In | 0.42 | 0.46 | Out | 0.23 | 0.22 |
| 6 | `22-investor-fomo-lunar-wireless` | Out | In | 0.39 | 0.47 | In | 0.62 | 0.61 |
| 7 | `77-sell-online-or-go-door-to-door` | In | Out | 0.16 | 0.27 | In | 0.57 | 0.56 |
| 8 | `24-unglue` | Out | In | 0.58 | 0.74 | In | 0.56 | 0.55 |
| 9 | `78-got-goals-grab-a-cru` | In | In | 0.65 | 0.68 | In | 0.62 | 0.58 |
| 10 | `37-can-tech-solve-americas-drug-crisis` | Out | In | 0.39 | 0.44 | In | 0.63 | 0.58 |
| 11 | `83-can-small-bras-be-a-big-market` | In | In | 0.35 | 0.43 | In | 0.56 | 0.49 |
| 12 | `38-a-beauty-industry-veteran-goes-rogue` | Out | In | 0.56 | 0.62 | In | 0.58 | 0.34 |
| 13 | `102-tether-bodyguard-of-the-grid` | In | In | 0.34 | 0.57 | In | 0.58 | 0.57 |
| 14 | `43-get-this-party-startup-in-here` | Out | Out | 0.18 | 0.62 | In | 0.56 | 0.54 |
| 15 | `104-dressd-red-carpet-or-red-ocean` | In | Out | 0.12 | 0.24 | In | 0.54 | 0.42 |
| 16 | `46-never-get-lost-again` | Out | Out | 0.01 | 0.08 | Out | 0.08 | 0.10 |
| 17 | `127-nectir-the-classroom-of-the-future` | In | In | 0.59 | 0.76 | In | 0.62 | 0.58 |
| 18 | `52-fitbit-for-dogs` | Out | In | 0.54 | 0.56 | In | 0.58 | 0.56 |
| 19 | `133-podcast-ai-josh-gets-displaced` | In | In | 0.58 | 0.64 | In | 0.55 | 0.48 |
| 20 | `54-is-selling-in-walmart-a-good-thing` | Out | In | 0.57 | 0.58 | In | 0.58 | 0.58 |
| 21 | `135-thoras-ai-the-twin-effect` | In | In | 0.47 | 0.46 | In | 0.58 | 0.57 |
| 22 | `61-can-a-zebra-survive-in-a-unicorn-world` | Out | In | 0.43 | 0.47 | Out | 0.20 | 0.22 |
| 23 | `143-vital-audio` | In | In | 0.64 | 0.73 | In | 0.62 | 0.62 |
| 24 | `62-pivot-or-die` | Out | FAIL | — | — | In | 0.56 | 0.54 |
| 25 | `148-esai` | In | In | 0.57 | 0.63 | In | 0.64 | 0.58 |
| 26 | `63-can-a-startup-solve-homelessness` | Out | In | 0.34 | 0.40 | Out | 0.20 | 0.24 |
| 27 | `149-dopl` | In | Out | 0.34 | 0.56 | In | 0.62 | 0.58 |
| 28 | `65-wait-your-app-does-what-exactly` | Out | Out | 0.14 | 0.39 | In | 0.64 | 0.61 |
| 29 | `152-oma-health` | In | In | 0.56 | 0.63 | In | 0.64 | 0.61 |
| 30 | `66-does-anyone-really-want-your-product` | Out | Out | 0.14 | 0.24 | In | 0.55 | 0.00 |
| 31 | `164-sundae-ltk-for-groceries` | In | In | 0.54 | 0.72 | FAIL | — | — |
| 32 | `69-when-less-is-not-more` | Out | In | 0.56 | 0.29 | Out | 0.18 | 0.24 |
| 33 | `170-original-sunshine-the-best-bagel-youve-never-heard-of` | In | In* | 0.62 | 0.68 | In | 0.62 | 0.58 |
| 34 | `70-what-is-michael-phelps-jamming-to-right-now` | Out | Out | 0.07 | 0.18 | FAIL | — | — |
| 35 | `172-my-town-ai-chatgpt-meets-simcity` | In | In | 0.55 | 0.76 | In | 0.62 | 0.58 |
| 36 | `71-raising-kids-is-hard-so-is-raising-a-startup` | Out | In | 0.52 | 0.61 | In | 0.57 | 0.44 |
| 37 | `175-above-health-the-allergy-clinic-of-the-future` | In | In | 0.58 | 0.66 | Out | 0.14 | 0.27 |

`*` Episode 170 has a usable endpoint but an invalid nested reference to rationale `R38`.

## Reproducibility and artifacts

All single-call prompts, manifests, selected precedents, raw provider responses, usage, cost, and runtime summaries are stored under `outputs/context-rich-single-llm-charles-through-final-in-2026-08-07/`. The agentic comparison source is `outputs/openrouter-luna-charles-two-iterations-semantic-full-2026-08-05/`.
