# Charles Hudson v2 full development run

## Executive finding

The two-phase Luna workflow now has complete, auditable outputs for all **86 evaluation-eligible Charles Hudson pitch-window decisions**: **17 Ins and 69 Outs** under the strict audited label policy. Every Phase 1 rationale artifact and Phase 2 decision artifact passed its frozen SHA-256 check. No episode was skipped.

The run succeeds operationally and produces rich, inspectable reasoning, but the raw decision heads are not yet reliable final classifiers. The any-check endpoint finds **14 of 17 Ins (82.4%)**, but also recommends an exploratory check for **56 of 69 Outs**. Its balanced accuracy is **50.6%**. The standard-check endpoint predicts every episode Out. For diamond search, any-check likelihood is the best raw signal, but only modestly exceeds prevalence: tie-aware average precision is **0.269** against **0.198** prevalence, with ROC AUC **0.565**.

The result is not “solve it by moving one threshold.” Three real Ins receive very low any-check likelihoods (0.27–0.30), while many Outs occupy 0.55–0.62. The outputs show two simultaneous errors:

1. Phase 2 often treats the absence of a fatal constraint as sufficient reason to buy a small option, creating many false Ins.
2. In three missed Ins, nonfatal portfolio or investor-domain concerns dominate strong founder, execution, traction, or mission evidence, creating false Outs.

The appropriate next step is a semantic rationale-core calibration model over the now-frozen Phase 1 outputs, evaluated with nested leave-one-out or other leakage-safe out-of-fold predictions. A targeted Phase 2 contract revision should separately distinguish a diligence condition from an automatic veto, but simply relaxing all constraints would worsen the already-high false-positive rate.

## Evaluation population

The label source is [`evaluation/labels/charles_pitch_window_decisions.json`](../../evaluation/labels/charles_pitch_window_decisions.json). The primary outcome is Charles's observed decision at the pitch-window cutoff, before later diligence or post-pitch information. This produces:

| Population | Episodes | In | Out |
|---|---:|---:|---:|
| Earlier Codex-comparison subset | 31 | 5 | 26 |
| Remaining eligible subset | 55 | 12 | 43 |
| **Full Charles development population** | **86** | **17** | **69** |

The 86 episodes are development data, not an untouched holdout. Thresholds or models selected after seeing these outcomes must be evaluated on a future locked set or by explicitly nested out-of-fold procedures.

### Conditional-interest sensitivity

Episodes 41 and 127 are measurement-sensitive. Charles says he is willing or mostly willing to invest provided a portfolio-conflict check clears, and later invests. The strict label file records both as Out because no unconditional commitment exists at the cutoff. For a broader “investment interest worth advancing” outcome, both reasonably count as In.

Under that broader policy, the population becomes 19 Ins and 67 Outs. The any-check head then has 16 true Ins, 54 false Ins, 13 true Outs, and 3 missed Ins; balanced accuracy is 51.8%, and any-check ranking AP is 0.331 against 0.221 prevalence. The scientific conclusion does not change, but ranking improves. Publication work should pre-register these as two distinct outcomes rather than silently switching labels.

## Locked execution method

Both phases used `openai/gpt-5.6-luna` through OpenRouter. The committed configuration is [`configs/openrouter-luna-charles-v2-batch.toml`](../../configs/openrouter-luna-charles-v2-batch.toml):

- pitch-only, leakage-audited current-company input;
- Charles's public-source investment memory and the rationale taxonomy;
- local `nomic-embed-text` retrieval through Ollama;
- up to three adaptive Phase 1 passes;
- up to two adaptive Phase 2 passes;
- no actual label, current full transcript, panel reaction, or post-pitch decision available to inference;
- at most two complete attempts per episode;
- raw model responses, retrievals, frozen outputs, validation findings, and token usage retained.

Phase 1 is **Rationale Investigation**. It asks investor-specific questions, retrieves exact wiki evidence, and locks pitch evidence, wiki evidence, taxonomy label, direction, salience, confidence, evidence status, constraint type, and severity. It is prohibited from deciding In or Out.

Phase 2 is **Investment Decision Synthesis**. It receives only the frozen Phase 1 investigation, assesses every rationale, constructs a risk ledger and opposing case, and produces separate any-check and standard-check endpoints plus 3–8 ordered deliberation steps.

The two cohorts are declared in [`charles-codex36-v2-development.json`](../../configs/batches/charles-codex36-v2-development.json) and [`charles-remaining55-v2-development.json`](../../configs/batches/charles-remaining55-v2-development.json). Authoritative batch manifests are:

- [`31-case batch status`](../../outputs/openrouter-luna-charles-v2-codex36-2026-08-02-fix1/batch-status.json)
- [`55-case batch status`](../../outputs/openrouter-luna-charles-v2-remaining55-2026-08-02/batch-status.json)

## Raw classification

| Endpoint / cohort | Cases | TP | FP | TN | FN | Balanced accuracy | In precision | In recall | In F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Any-check, earlier 31 | 31 | 5 | 22 | 4 | 0 | 57.7% | 18.5% | 100.0% | 31.3% |
| Any-check, remaining 55 | 55 | 9 | 34 | 9 | 3 | 48.0% | 20.9% | 75.0% | 32.7% |
| **Any-check, full 86** | **86** | **14** | **56** | **13** | **3** | **50.6%** | **20.0%** | **82.4%** | **32.2%** |
| **Standard-check, full 86** | **86** | **0** | **0** | **69** | **17** | **50.0%** | **0.0%** | **0.0%** | **0.0%** |

Overall accuracy is 31.4% for any-check and 80.2% for standard-check, but both figures are misleading under the 80.2% Out prevalence. Balanced accuracy and the confusion counts show that neither raw endpoint is a useful final classifier at its current fixed decision rule.

For context, the historical GPT-5.5/Codex method on the same earlier 31 revised labels produced 3 true Ins, 2 false Ins, 24 true Outs, and 2 missed Ins—76.2% balanced accuracy. That earlier method is materially stronger as a raw classifier on that subset, although the v2 workflow provides much richer, versioned rationale and deliberation artifacts. This comparison is developmental and does not establish generalization for either method.

## Diamond-search ranking

| Raw score | ROC AUC | Tie-aware AP | Baseline prevalence | Interpretation |
|---|---:|---:|---:|---|
| Any-check likelihood | **0.565** | **0.269** | 0.198 | Best raw v2 ranking signal, but modest |
| Standard-check likelihood | 0.564 | 0.231 | 0.198 | Slightly above prevalence |
| Model `ranking_score` | 0.443 | 0.186 | 0.198 | Worse than prevalence; not a diamond score |

Many episodes share the same likelihood, so arbitrary top-k tie breaking can materially change a shortlist. Threshold-defined review queues are more honest:

| Any-check likelihood cutoff | Pitches reviewed | Actual Ins found | Precision | Recall |
|---:|---:|---:|---:|---:|
| >= 0.62 | 17 | 6 | 35.3% | 35.3% |
| >= 0.61 | 20 | 7 | 35.0% | 41.2% |
| >= 0.60 | 25 | 8 | 32.0% | 47.1% |
| >= 0.58 | 39 | 9 | 23.1% | 52.9% |
| >= 0.55 | 66 | 13 | 19.7% | 76.5% |

Reviewing the 20 pitches at or above 0.61 finds 7 of the 17 Ins. Precision is 1.77 times the 19.8% base rate, but 10 Ins remain outside the queue. This is a weak-to-moderate prioritization signal, not yet a “needle finder” result.

The model-generated `ranking_score` is inversely useful at the top of the list: the 14 pitches scoring at least 0.80 contain no Ins. Its text and behavior indicate that it often measures **human-review urgency, unresolved complexity, or diligence importance**, not investment propensity. It must be renamed or redesigned; it should not be presented as the diamond-search endpoint.

## Threshold analysis

The best balanced-accuracy cutoff selected on the full development data is any-check likelihood >= 0.60:

- 8 true Ins, 17 false Ins, 52 true Outs, and 9 missed Ins;
- 61.2% balanced accuracy;
- 32.0% In precision, 47.1% In recall, and 38.1% In F1.

This is descriptive, post-hoc development analysis—not a valid performance estimate. It shows that a more conservative threshold can reduce false positives, but it does not solve the low-scoring real Ins. A global threshold therefore cannot be the only intervention.

Decision confidence must not be ranked as if it were In probability. It is confidence in whichever verdict was made; a confident Out can have high confidence. Confidence can be a feature in a calibrated downstream model only when paired with decision direction and likelihood.

## Error analysis

### The three missed Ins

#### 83 — Pepper

Charles explicitly offered $25K. Phase 1 successfully recovered the customer pain, custom product design, Urban Outfitters reorder, press/search demand, reported $25 CAC and $130 LTV, 71% margin, founder-market connection, and category-expansion thesis. Phase 2 nevertheless returned Out at 0.30 because it treated the unresolved True & Co portfolio overlap as a controlling gate and added round/market/economics uncertainty.

This is principally a weighting error, not a rationale-discovery failure. The actual behavior demonstrates that Charles could make a small investment while conflict diligence remained. A secondary information limitation is that the current pitch-only input excludes supportive panel discussion that changed Charles's market interpretation.

#### 143 — Vital Audio

Charles offered $25K and said he was about 90% sure the relevant portfolio relationship was complementary, subject to checking. Phase 1 recovered founder authenticity, clinical and technical expertise, a functioning system, reported 96% accuracy, studies with more than 500 participants, patents, FDA interaction, five initial partners, reported contract revenue, market breadth, and capital efficiency.

Phase 2 returned Out at 0.27 because a wiki statement about Charles's limited ability to opine on healthcare became a material investor-fit conflict. This overgeneralizes a public principle into a target-specific veto. Charles's actual small check shows that founder expertise and a bounded commitment could overcome his personal domain limitation.

#### 152 — OMA Health

Charles made an unconditional $25K commitment, with diligence applying only to a possible increase. Phase 1 recovered the founder's payer/provider experience, authentic mission, a concrete credentialing problem, a large underserved population, a platform vision, an 8–12% take rate, beta practitioners, and a fund-compatible $1M pre-seed round.

Phase 2 returned Out at 0.28. It treated the pre-MVP stage and healthcare-domain limitation as controlling while requiring signed payer outcomes, proven cohorts, unit economics, and defensibility. The model applied a mature-evidence bar that is inconsistent with an exploratory pre-seed check and underweighted the unusually strong founder-problem-mission fit that appears to have driven Charles's exception.

### Why there are 56 false Ins

The common false-positive pattern is nearly the mirror image. Phase 2 correctly treats missing evidence as uncertainty rather than fabricated adverse evidence, but then frequently concludes that any nonfatal opportunity deserves a bounded exploratory check. “No fatal constraint,” a plausible founder, and some early de-risking become sufficient for optionality even when company-specific evidence does not clear Charles's opportunity-cost bar.

The current prompt also contains a tension: it says minor/routine conditions can support an any-check In, but separately says an unresolved portfolio conflict is material and normally blocks any-check. This makes portfolio and domain concerns too rigid in some cases while the general optionality rule remains too permissive elsewhere.

## Recommended solution

### 1. Preserve Phase 1; learn the decision boundary from its semantics

The full run now provides 1,409 frozen rationales—an average of 16.4 per episode—with evidence, taxonomy labels, direction, salience, confidence, evidence status, constraint type, severity, deal context, and provenance. These are the appropriate inputs for a **semantic rationale-core model**.

The downstream model should use compact, interpretable features such as:

- positive/negative primary-rationale counts and confidence mass by taxonomy label;
- affirmative evidence versus missing evidence;
- constraint kind and severity;
- founder, market, execution, economics, vision, fund-fit, and portfolio interactions;
- any-check and standard-check likelihoods as model opinions, not final truth;
- accepted/provisional status and selected process-quality findings.

For each episode, train only on the other episodes and generate an out-of-fold prediction. Hyperparameters, feature selection, and thresholds must be chosen inside the training fold. Report both pitch classification and ranking metrics. With only 17 strict Ins, use strong regularization and treat results as development evidence until a locked holdout exists.

This can be done without rerunning Phase 1. The rationale artifacts are frozen and hash-verified.

### 2. Separate a diligence condition from a veto

The Phase 2 contract should explicitly distinguish:

- fatal prohibition;
- target-specific affirmative adverse risk;
- commitment conditional on routine diligence;
- size-limiting uncertainty;
- ordinary noted diligence item;
- missing evidence.

Only a fatal prohibition should automatically force any-check Out. A material but nonfatal concern should lower likelihood and require a named condition; it should not automatically block or automatically permit a check. This change directly addresses Pepper and Vital Audio, but must be tested against matched Out controls before replaying all Phase 2 outputs, because a blanket relaxation would worsen false positives.

### 3. Redefine the two outcome tasks

Maintain two pre-registered labels:

1. **Current capital commitment**: strict observed In/Out at the pitch cutoff.
2. **Advance / investment interest**: includes explicit conditional willingness pending routine diligence.

The first supports pitch-decision classification. The second better matches human-in-the-loop prioritization. Episodes 41 and 127 demonstrate why these should not be forced into one ambiguous label.

### 4. Retire the current `ranking_score` as a diamond endpoint

Rename it `human_review_priority` if its present meaning is retained. Build diamond ranking from calibrated In propensity or a separately specified expected-value score. Do not combine review urgency and investment propensity in one field.

## Reasoning-flow example: episode 77

Episode 77 is an actual In that the v2 any-check endpoint correctly predicts In at 0.61, while standard-check remains Out at 0.31. Its selected artifacts are under [`77-sell-online-or-go-door-to-door`](../../outputs/openrouter-luna-charles-v2-codex36-2026-08-02-fix1/attempt-1/77-sell-online-or-go-door-to-door).

Phase 1 uses two passes to lock positive evidence about the market problem, product appeal, unit economics, founder execution, vision alignment, mission, and right-sized check optionality. It also preserves unresolved or negative evidence about repeatability, competition, defensibility, growth projections, and routine portfolio adjacency.

Phase 2 then produces separate continuous tracks. The any-check track argues that a bounded investment can buy information while limiting exposure and ends at In 0.61. The standard-check track weighs the same unresolved scale, moat, economics, and terms issues under a stricter bar and ends at Out 0.31. The six deliberation steps make the movement, counterargument, controlling rationales, and reversal conditions inspectable. This is the intended value of the two-phase architecture even though aggregate calibration remains weak.

## Operational reliability and usage

| Item | Result |
|---|---:|
| Usable decisions | 86 / 86 |
| Selected attempt 1 | 83 |
| Selected attempt 2 | 3 |
| Phase 1 accepted / provisional | 27 / 59 |
| Phase 2 accepted / provisional | 55 / 31 |
| Phase 1 iterations: 1 / 2 / 3 | 52 / 22 / 12 |
| Phase 2 iterations: 1 / 2 | 22 / 64 |
| Frozen rationales | 1,409 |
| Deliberation steps | 552 |

The selected successful artifacts used 6,673,039 input tokens, 48,404 cached input tokens, and 1,357,383 output tokens, costing **$1.64287** through OpenRouter. Three discarded first attempts consumed an additional 368,580 input and 51,316 output tokens, costing **$0.07686**. Earlier schema-debugging attempts and one interrupted call are engineering overhead and are not included in the selected-artifact figure.

Operational completion does not mean every output fully converged. Phase 1 is provisional in 59 cases and Phase 2 in 31. The most common process findings are planner fallback (47 episodes), risk-ledger coverage mismatch (26), lack of retrieval novelty (10), and normalized untyped constraints. These findings do not erase the frozen decisions, but they must be disclosed and should become quality-improvement targets before a publication run.

## Conclusion

Charles is now complete at the artifact level: all 86 eligible pitch-only cases have frozen rationale investigations, decision syntheses, per-turn traces, usage, and verified manifests. The full run supports the architecture's value as an auditable reasoning instrument, but it does not support claiming that the raw Luna endpoint is a strong classifier or diamond-search system.

The strongest scientific next move is to treat the agent outputs as structured measurements of investor-specific reasoning and learn the final decision boundary from those measurements using leakage-safe out-of-fold evaluation. In parallel, the contract should distinguish conditional diligence from vetoes and separate current commitment from advance-worthy interest. This preserves the rich reasoning while directly targeting the two observed calibration failures.

## Appendix: all 86 episodes

`A` means accepted and `P` provisional. `P1/P2 n` is the number of selected passes; `Try` is the selected full attempt.

| Episode | Actual | Any | Any p | Standard | Standard p | Review | P1 | P1 n | P2 | P2 n | Try |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 18-rowvigor | Out | In | 0.58 | Out | 0.24 | 0.72 | A | 2 | A | 2 | 1 |
| 20-harper-wilde | Out | Out | 0.30 | Out | 0.15 | 0.82 | P | 1 | A | 2 | 1 |
| 22-investor-fomo-lunar-wireless | Out | In | 0.61 | Out | 0.27 | 0.78 | A | 1 | A | 1 | 1 |
| 24-unglue | Out | In | 0.55 | Out | 0.27 | 0.80 | P | 1 | A | 2 | 1 |
| 37-can-tech-solve-americas-drug-crisis | Out | In | 0.56 | Out | 0.25 | 0.62 | P | 1 | P | 2 | 1 |
| 38-a-beauty-industry-veteran-goes-rogue | Out | In | 0.58 | Out | 0.22 | 0.64 | A | 1 | A | 1 | 1 |
| 39-this-pitch-is-damn-near-perfect | In | In | 0.62 | Out | 0.32 | 0.78 | P | 1 | P | 2 | 1 |
| 41-can-this-startup-help-retailers-take-on-amazon | Out | In | 0.56 | Out | 0.27 | 0.67 | A | 1 | A | 2 | 1 |
| 43-get-this-party-startup-in-here | Out | In | 0.58 | Out | 0.34 | 0.63 | P | 3 | A | 1 | 1 |
| 46-never-get-lost-again | Out | Out | 0.28 | Out | 0.10 | 0.48 | P | 1 | A | 2 | 1 |
| 52-fitbit-for-dogs | Out | In | 0.60 | Out | 0.34 | 0.72 | P | 1 | P | 2 | 2 |
| 54-is-selling-in-walmart-a-good-thing | Out | Out | 0.25 | Out | 0.18 | 0.78 | A | 1 | A | 1 | 1 |
| 61-can-a-zebra-survive-in-a-unicorn-world | Out | In | 0.54 | Out | 0.25 | 0.78 | P | 1 | P | 2 | 1 |
| 62-pivot-or-die | Out | In | 0.55 | Out | 0.29 | 0.82 | A | 1 | A | 2 | 1 |
| 63-can-a-startup-solve-homelessness | Out | Out | 0.43 | Out | 0.15 | 0.86 | A | 1 | A | 2 | 1 |
| 65-wait-your-app-does-what-exactly | Out | In | 0.58 | Out | 0.32 | 0.72 | A | 2 | A | 1 | 1 |
| 66-does-anyone-really-want-your-product | Out | In | 0.54 | Out | 0.34 | 0.68 | P | 1 | A | 2 | 1 |
| 68-dogs-dating-a-match-made-in-heaven | In | In | 0.58 | Out | 0.28 | 0.64 | P | 2 | A | 1 | 1 |
| 69-when-less-is-not-more | Out | In | 0.62 | Out | 0.24 | 0.78 | A | 1 | A | 2 | 1 |
| 70-what-is-michael-phelps-jamming-to-right-now | Out | In | 0.55 | Out | 0.20 | 0.72 | A | 1 | A | 2 | 1 |
| 71-raising-kids-is-hard-so-is-raising-a-startup | Out | In | 0.58 | Out | 0.25 | 0.68 | P | 1 | A | 1 | 1 |
| 72-how-niche-is-too-niche | Out | Out | 0.35 | Out | 0.18 | 0.86 | A | 1 | A | 2 | 1 |
| 73-enough-sales-calls-already | Out | In | 0.62 | Out | 0.28 | 0.72 | P | 2 | A | 1 | 1 |
| 74-stop-killing-your-plants | Out | In | 0.56 | Out | 0.43 | 0.72 | P | 1 | A | 2 | 1 |
| 75-i-want-bro-money-too | Out | In | 0.56 | Out | 0.22 | 0.82 | P | 1 | P | 2 | 1 |
| 77-sell-online-or-go-door-to-door | In | In | 0.61 | Out | 0.31 | 0.78 | P | 2 | P | 2 | 1 |
| 78-got-goals-grab-a-cru | In | In | 0.62 | Out | 0.30 | 0.70 | A | 2 | A | 1 | 1 |
| 79-the-anti-juul | Out | Out | 0.32 | Out | 0.18 | 0.78 | A | 1 | A | 2 | 1 |
| 80-can-this-app-silence-the-trolls | Out | In | 0.60 | Out | 0.32 | 0.74 | P | 1 | A | 1 | 1 |
| 82-a-deal-too-perfect-to-pass-up | Out | Out | 0.24 | Out | 0.10 | 0.82 | A | 1 | A | 2 | 1 |
| 83-can-small-bras-be-a-big-market | In | Out | 0.30 | Out | 0.18 | 0.62 | A | 1 | A | 2 | 1 |
| 84-the-napkin-pitch | Out | In | 0.56 | Out | 0.26 | 0.64 | P | 1 | P | 2 | 1 |
| 85-will-coronavirus-kill-this-deal | Out | In | 0.56 | Out | 0.30 | 0.82 | A | 1 | A | 1 | 1 |
| 86-if-the-suit-fits-invest | Out | In | 0.66 | Out | 0.31 | 0.80 | P | 1 | P | 2 | 1 |
| 87-uber-for-pets | Out | In | 0.62 | Out | 0.27 | 0.58 | P | 1 | P | 2 | 1 |
| 89-a-big-bet-on-an-ex-trucker | Out | In | 0.54 | Out | 0.27 | 0.72 | P | 1 | A | 2 | 1 |
| 102-tether-bodyguard-of-the-grid | In | In | 0.56 | Out | 0.29 | 0.76 | P | 2 | A | 2 | 1 |
| 104-dressd-red-carpet-or-red-ocean | In | In | 0.56 | Out | 0.25 | 0.76 | A | 1 | P | 2 | 1 |
| 105-teffola-granola-guy-approved | Out | In | 0.58 | Out | 0.28 | 0.78 | P | 2 | A | 2 | 1 |
| 106-the-calendar-wars | Out | In | 0.55 | Out | 0.26 | 0.75 | P | 3 | P | 2 | 1 |
| 107-terrascope-a-computer-vision-tale | Out | In | 0.57 | Out | 0.30 | 0.78 | P | 3 | P | 2 | 1 |
| 110-handsdown-the-accelerator-pitch | Out | In | 0.58 | Out | 0.27 | 0.68 | P | 1 | P | 2 | 1 |
| 111-stigma-whose-business-is-it-anyway | Out | In | 0.62 | Out | 0.28 | 0.78 | P | 1 | P | 2 | 1 |
| 114-brevity-your-ai-pitch-coach | Out | In | 0.58 | Out | 0.24 | 0.62 | A | 1 | A | 2 | 1 |
| 115-gemist-the-crown-jewel-of-venture | Out | In | 0.58 | Out | 0.34 | 0.76 | P | 2 | A | 2 | 2 |
| 117-lotus-hardware-hail-mary | Out | In | 0.58 | Out | 0.30 | 0.78 | P | 1 | A | 1 | 1 |
| 118-uncovered-the-business-of-true-crime | Out | In | 0.55 | Out | 0.30 | 0.70 | P | 3 | P | 2 | 1 |
| 120-bevz-a-tech-bro-walks-into-a-corner-store | Out | In | 0.62 | Out | 0.28 | 0.78 | P | 1 | P | 2 | 1 |
| 122-wist-the-killer-use-case-for-vr | Out | In | 0.57 | Out | 0.25 | 0.62 | P | 3 | A | 1 | 1 |
| 124-ecgo-turning-trash-into-treasure-with-ai | Out | In | 0.59 | Out | 0.28 | 0.74 | A | 2 | P | 2 | 1 |
| 126-handle-the-uncut-pitch | Out | Out | 0.22 | Out | 0.09 | 0.78 | P | 2 | A | 2 | 1 |
| 127-nectir-the-classroom-of-the-future | Out | In | 0.64 | Out | 0.34 | 0.72 | P | 2 | A | 2 | 2 |
| 129-reddrop-pitching-the-big-vision | Out | Out | 0.35 | Out | 0.18 | 0.82 | P | 1 | P | 2 | 1 |
| 131-flock-ai-model-generated-models | Out | Out | 0.22 | Out | 0.10 | 0.86 | A | 1 | P | 2 | 1 |
| 133-podcast-ai-josh-gets-displaced | In | In | 0.55 | Out | 0.29 | 0.74 | P | 3 | P | 2 | 1 |
| 135-thoras-ai-the-twin-effect | In | In | 0.60 | Out | 0.34 | 0.76 | P | 3 | A | 2 | 1 |
| 136-kredfeed-the-next-mexican-unicorn | Out | In | 0.57 | Out | 0.34 | 0.65 | A | 1 | A | 2 | 1 |
| 139-futuremoney | Out | In | 0.62 | Out | 0.28 | 0.78 | P | 3 | A | 1 | 1 |
| 140-gecko-materials | Out | Out | 0.34 | Out | 0.22 | 0.64 | A | 2 | A | 2 | 1 |
| 143-vital-audio | In | Out | 0.27 | Out | 0.12 | 0.36 | A | 2 | P | 2 | 1 |
| 144-bridgefy | Out | In | 0.56 | Out | 0.29 | 0.72 | P | 1 | P | 2 | 1 |
| 145-immersioned | Out | Out | 0.34 | Out | 0.18 | 0.72 | P | 3 | A | 2 | 1 |
| 146-recraft-beer | Out | In | 0.60 | Out | 0.27 | 0.76 | P | 1 | P | 2 | 1 |
| 148-esai | In | In | 0.62 | Out | 0.28 | 0.76 | P | 2 | A | 1 | 1 |
| 149-dopl | In | In | 0.53 | Out | 0.22 | 0.72 | P | 1 | P | 2 | 1 |
| 151-feedback-intelligence | Out | In | 0.58 | Out | 0.24 | 0.72 | A | 1 | A | 1 | 1 |
| 152-oma-health | In | Out | 0.28 | Out | 0.08 | 0.70 | P | 1 | A | 1 | 1 |
| 156-barberinos-italian-luxury-takes-on-the-american-dream | Out | In | 0.56 | Out | 0.24 | 0.70 | P | 1 | A | 2 | 1 |
| 158-avelo-the-worlds-first-smart-running-shoe | Out | In | 0.57 | Out | 0.24 | 0.62 | A | 1 | A | 2 | 1 |
| 159-aura-finance-chasing-the-ghost-of-mint | Out | In | 0.61 | Out | 0.37 | 0.68 | P | 2 | P | 2 | 1 |
| 162-jungle-your-ai-study-coach | Out | In | 0.56 | Out | 0.30 | 0.76 | P | 1 | P | 2 | 1 |
| 163-curiedx-ai-pocket-doctor | Out | Out | 0.18 | Out | 0.06 | 0.86 | P | 2 | A | 2 | 1 |
| 164-sundae-ltk-for-groceries | In | In | 0.55 | Out | 0.30 | 0.77 | P | 3 | P | 2 | 1 |
| 166-awear-whoop-for-your-brain | Out | In | 0.56 | Out | 0.25 | 0.71 | P | 3 | P | 2 | 1 |
| 167-levee-cleaning-up-hotels-dirty-secret | Out | In | 0.60 | Out | 0.28 | 0.72 | P | 2 | A | 1 | 1 |
| 168-finnecto-where-deals-dont-die | Out | In | 0.62 | Out | 0.35 | 0.72 | P | 1 | P | 2 | 1 |
| 169-mappa-can-voice-ai-find-you-a-job-or-a-date | Out | In | 0.62 | Out | 0.28 | 0.78 | P | 1 | A | 1 | 1 |
| 170-original-sunshine-the-best-bagel-youve-never-heard-of | In | In | 0.62 | Out | 0.35 | 0.78 | A | 1 | A | 1 | 1 |
| 171-doctours-vc-funded-hair-transplants | Out | In | 0.58 | Out | 0.24 | 0.76 | P | 1 | A | 2 | 1 |
| 172-my-town-ai-chatgpt-meets-simcity | In | In | 0.63 | Out | 0.27 | 0.65 | P | 1 | P | 2 | 1 |
| 174-investrio-quickbooks-for-the-people | Out | In | 0.62 | Out | 0.27 | 0.82 | P | 2 | A | 1 | 1 |
| 175-above-health-the-allergy-clinic-of-the-future | In | In | 0.62 | Out | 0.25 | 0.78 | P | 3 | A | 1 | 1 |
| 176-your-ai-website-sucks-peachweb-is-the-future | Out | In | 0.55 | Out | 0.30 | 0.74 | P | 2 | A | 2 | 1 |
| 178-cosmicbrain-ai-how-to-train-your-robot | Out | In | 0.55 | Out | 0.24 | 0.63 | A | 1 | A | 2 | 1 |
| 179-minimis-declaring-war-on-garmin | Out | In | 0.56 | Out | 0.20 | 0.84 | P | 2 | P | 2 | 1 |
| gnara-disrupting-your-pants | Out | In | 0.56 | Out | 0.30 | 0.71 | P | 2 | A | 2 | 1 |
