# Current Assessment Error Analysis

Date: 2026-10-20. Cohort: `all-previous-current-code-v1`.

## Conclusion

The pipeline is not yet demonstrated reliable for predicting a particular investor's in-room commitment. This is a development regression cohort, not an untouched benchmark. The prior 63.8% contains known leakage exposure and multiple configurations; comparing it with this cohort does not establish that the current changes helped or harmed clean prediction. Current wiki lineage gaps also remain unresolved. Do not label 57.4% leakage-clean.

## Measured Results

- 94 completed, 1 failed; 19 investors, 63 distinct episodes; 93 completed A inputs and 1 B input. The same episode across investors and the A/B pair are correlated observations.
- Correct: 54/94 (57.4%). Always predicting Out would get 53/94 (56.4%). This is a descriptive majority baseline, not proof that the model has no useful signal.
- Actual In: 21/41 (51.2%). Actual Out: 33/53 (62.3%). Balanced accuracy: 56.7%.
- 20 false Ins and 20 false Outs. A global stricter/looser instruction cannot address both mechanisms.
- Accepted: 20/32 correct (62.5%); provisional: 34/62 (54.8%). Accepted is workflow validation, not verified prediction correctness.
- 34 stopped at corpus exhaustion; 28 at iteration limit. Increasing iteration count is not evidence of a remedy.
- Completed cost $11.4405812; failed case recorded $0.0626552; total saved usage $11.5032364.

## Evidence And Limits

Reviewed all 40 error cases' saved decision justifications against decisive candidate reference excerpts; inspected contrasting correct cases and implementation of commitment/exception checks. All 115 decisive excerpts for these errors were found as normalized substrings in the original transcripts. This establishes source quotation, not validity of the taxonomy, causal importance, or temporal scope. Manually inspected original dialogue for Lotus negotiation, Handle personal-interest wording, Rohit's prior-round context and Sarah's angel/debrief distinction. No model reruns, inference changes, label changes or paid judging were performed.

The companion JSON records all 94 decisions, candidate references, stage diagnostics, history availability and artifact paths. The appendix records a review hypothesis for every error. These are not independently validated causal labels. Controlled tests are still needed to attribute gains to a particular intervention.

## Findings

### 1. Prediction target and available information sometimes differ

**Paige/Lotus is confirmed:** the supplied founders-only pitch finishes at a $14m post-money cap with flexibility on terms. The original discussion goes to offers at $6m, a counter at $8m, temporary passes, and acceptance at $6m. The model's Out on the initial offer is scored against the eventual negotiated In. Both are coherent for different offers. Restoring the acceptance would leak the answer; do not do that. Preserve the in-room outcome, and separately identify whether it is comparable with the supplied pre-decision offer. Negotiation outcomes require their own task or explicit uncertainty.

**Information unavailable to the model matters:** Rohit/Supercool cites having missed the previous round; Mac/Tether explicitly relies on other investors' expertise (A excludes their comments). Such cases cannot be repaired by inventing evidence or returning target speech to the input. Distinguish predictable evidence from private preference, relationship and interaction effects.

**Commitment and vehicle need consistent definitions:** Paige/Handle says she would be interested in a small personal check and is happy to chat, while the model asserts a concrete commitment. Other cases include partner approval, conflict clearance, product testing or lead-investor conditions. Conditional commitment is not automatically Out, and interest is not automatically In. These boundaries must be adjudicated consistently, with fund/personal/unknown vehicle recorded separately.

### 2. Investor-specific weighting is weak in both directions

False Ins commonly convert real concerns into ordinary diligence because the founders or traction look attractive: PaySquad across Charles, Elizabeth, Jesse and Mike; RideIQ; PodcastAI; Gemist. False Outs apply a generic proof/economics bar where the investor made a founder, mission or personal bet: Cru, Shipsi, Sarah's Feedback/Spot On, Sheel's Hava. This is not primarily missing arithmetic: several outputs compute the revenue bridge correctly and then weight it differently from the investor.

Jesse/Jungle versus Jesse/Ventrix illustrates inconsistent application: a general moat preference defeats demonstrated growth in Jungle, while a founder exception supports unfamiliar hardware in Ventrix. The observed decisions are the reverse. Useful analogies need investor role, vehicle, stage, decisive risk and evidence of tolerated risk, not simply similar positive attributes.

### 3. Existing safeguards are structural and self-assessed

`commitment_findings` binds positive pitch rationales and investor evidence IDs. Its risk-support checks apply only when the model chooses `founder_conviction_exception`. The model also supplies the support booleans itself. This prevents certain malformed/unsupported records; it does not independently judge whether an analogy warrants a commitment.

25 of 31 conventional-fit Ins explicitly say a material difference defeats the exception; 14 of those 25 are wrong. This is a review signal, not a rule for flipping them: 11 are correct, and an independent conventional case can legitimately succeed after rejecting an exception. RideIQ is an example where the final explanation still relies on broad tolerance analogies and treats recognized scale/valuation objections as diligence. Simply forcing all these cases Out would create further errors.

### 4. History coverage is sometimes a hard information limit

For Paige/Lotus the accessible corpus has 7 Outs, 4 unobserved and zero Ins. For Sarah/Feedback it has four Outs and zero Ins. Neither is just failure to retrieve an available positive case. Only two of the 20 false Outs have zero accessible Ins, so this does not explain the whole error set. Some correct Jenny and Nicole predictions also have no positive history; absence of positive precedent must not become an automatic rejection.

### 5. Rationale scoring is not yet a uniform validated benchmark

35 completed cases use `live_founder_present_v2`; 59 retain `original_pitch_only`, whose original extraction included immediate post-founder debrief. All references are automated candidates, not human-validated ground truth. Do not pool these protocols as a single rationale score.

On the 35 newer-scope cases, exact-label micro F1 is 35.7%; label-plus-direction F1 is 18.3%. On the 59 older-scope cases they are 32.6% and 16.5%. This signals a substantial mismatch, but exact taxonomy matching can penalize synonyms and predictions of possible concerns that the investor never voiced. We must distinguish observed rationale reconstruction from plausible ex-ante rationale prediction.

37/40 errors miss at least one candidate decisive label; 10 have direction disagreements; 9 detect a decisive label without controlling the decision; 8 include a decisive label among controlling reasons but still get the decision wrong. These categories overlap. Crucially, 46/54 correct decisions also miss a decisive label: decision accuracy alone does not establish rationale fidelity.

### 6. The failed case has an underlying generation/validation failure

Beck/RedDrop never produced a valid Phase 1 investigation across four turns. Recorded issues include overlong retrieval questions, inaccessible evidence IDs leaving a constraint unsupported, and duplicate material-statement pitch IDs. The runner then tried to load the nonexistent investigation and surfaced FileNotFoundError. That is a secondary reporting failure, not the original cause. Preserve the raw invalid responses and report the phase failure directly; do not turn invalid output into a scored prediction.

## Stable Plan Before Another Broad Run

1. **Freeze the current 95-case manifest, inputs, labels, outputs and this analysis as development evidence.** Keep the two excluded Mac cases separately. Never advertise reused examples as fresh validation.
2. **Freeze one evaluation contract:** in-room commitment vs later funding; affirmative conditional commitment vs interest; vehicle; offer/negotiation snapshot; source boundaries. Adjudicate uncertainty without changing labels to match model outputs. Keep raw full-cohort score and comparability/uncertainty strata side by side.
3. **Finish data validity first:** complete live-founder reference review and semantic rationale checks; complete target-specific wiki lineage and aggregate filtering; verify founder-only context without restoring target opinions/negotiation outcomes. Future sources remain allowed when independent of the target decision. No blanket date cutoff.
4. **Make one coherent inference design change, tested in isolation:** represent observed investor tolerances and priorities by role/vehicle; distinguish missing knowledge from a negative preference; independently check claim/evidence and analogy support across every decision route. Output evidence sufficiency/uncertainty separately from the required binary prediction. Do not grant correctness merely because a judge agrees.
5. **Evaluate fixed ablations on development data:** current baseline; data/protocol corrections only; investor-calibration change only; combined. Measure decision, direction-aware rationale fidelity, coverage, cost and stability. Keep settings and seeds where supported recorded; repeated runs measure variation, not a chance to select favorable output.
6. **Select once, then test once on genuinely untouched cases grouped by episode/company.** Ensure target information is excluded from every case's wiki/history and from any derived training/calibration summaries. Report A and B separately, per-VC results and both error directions. Agree success thresholds before opening holdout outcomes; require meaningful improvement over simple baselines, acceptable In/Out recall and rationale fidelity, not just a better pooled percentage. If no adequate untouched set remains, collect more or state that generalization is unmeasured.
7. **Stop rule:** if the frozen validation fails, do not launch another full prompt-tweaked batch. Reassess whether the available evidence can support investor-specific prediction; retain an evidence report/uncertainty output rather than claiming a reliable decision clone.

These are recommendations, not changes applied by this analysis. No additional paid execution was started.

## Error-By-Error Review

Each row is an evidence-based review hypothesis, not proof of a unique root cause. Actual labels remain unchanged.

| VC | Episode | Actual → Model | Review finding |
|---|---|---|---|
| beck-bamberger-bad-ideas-group | 121-ride-iq-peloton-for-horses | Out → In | Scale/vision concern acknowledged but outweighed by generic traction and unrelated category-tolerance precedents. |
| charles-hudson-precursor-ventures | 187-paysquad-buy-now-pay-together | Out → In | Payments throughput and retained economics downgraded to diligence despite being the observed reason to pass. |
| charles-hudson-precursor-ventures | 188-church-ai-deploying-gods-agents | Out → In | Required church ACV and venture-scale adoption discounted in favor of founder fit and usage. |
| charles-hudson-precursor-ventures | 189-wonderade-juice-protein-or-a-brand | Out → In | Founder credentials outweigh the observed parent/child positioning and entrenched juice-belief objection. |
| charles-hudson-precursor-ventures | 41-can-this-startup-help-retailers-take-on-amazon | In → Out | Unproven economics and forecasting discipline outweigh the observed logistics-API thesis; actual commitment also has conflict clearance condition. |
| cyan-banister-long-journey-ventures | 127-nectir-the-classroom-of-the-future | In → Out | One institutional-education Out generalized into a category preference despite strong current adoption; actual investment subject to partner discussion. |
| cyan-banister-long-journey-ventures | 140-gecko-materials | Out → In | Technical validation overrides software focus and unresolved diligence; small-check framing does not match observed refusal even as angel. |
| cyan-banister-long-journey-ventures | 141-christies-chips | Out → In | Traction overrides CPG/vehicle fit; actual response postpones decision pending discussion with spouse. |
| elizabeth-yin-hustle-fund | 184-ventrix-the-gas-plant-of-the-future | In → Out | Hardware capital efficiency dominates; reference instead emphasizes founder learning, European financing support and investability at the right price. |
| elizabeth-yin-hustle-fund | 185-its-electric-beating-tesla-was-the-easy-part | Out → In | Deployment/RFP traction outweighs first-institutional-check preference, dilution and municipal execution risk. |
| elizabeth-yin-hustle-fund | 187-paysquad-buy-now-pay-together | Out → In | Founder/merchant signals outweigh enterprise-sales-cycle and obtainable payments-revenue concerns. |
| elizabeth-yin-hustle-fund | 78-got-goals-grab-a-cru | In → Out | Missing paid conversion and mature economics outweigh founder bet, mission and audience-monetization thesis. |
| howie-diamond-pure-ventures | 4-skylights | Out → In | Operating trials treated as enough proof while observed investor explicitly calls company too early. |
| howie-diamond-pure-ventures | 48-will-this-space-startup-make-it-off-the-ground | In → Out | Missing technical demonstration outweighs observed excitement and ability to contribute space-domain expertise. |
| jake-chapman-marque-ventures | 4-skylights | Out → In | Functioning deployment and team outweigh weak incremental customer need and incumbent entry risk. |
| jesse-middleton-flybridge | 148-esai | In → Out | Monetization proof dominates observed product appeal, focused distribution and analogous portfolio experience. |
| jesse-middleton-flybridge | 162-jungle-your-ai-study-coach | In → Out | General structural-moat preference and repeatability requirements dominate observed ROAS and early growth evidence; original reference includes debrief. |
| jesse-middleton-flybridge | 184-ventrix-the-gas-plant-of-the-future | Out → In | Hardware founder exception overrides observed lack of category knowledge and willingness to commit. |
| jesse-middleton-flybridge | 187-paysquad-buy-now-pay-together | Out → In | Founder exception outweighs thin payments economics and competitive/early-stage concerns. |
| jillian-manus-structure-capital | 1-babyscripts | In → Out | Valuation multiple and missing economics dominate observed personal connection and investor ability to add value. |
| jillian-manus-structure-capital | 3-industrial-organic | Out → In | Mission and founder-market fit outweigh unrealistic burn and need for operating milestones. |
| mac-conwell | 102-tether-bodyguard-of-the-grid | In → Out | Pre-product status and economics dominate problem/price attractiveness; actual investor also relied on other panelists expertise, absent from A. |
| mac-conwell | 111-stigma-whose-business-is-it-anyway | In → Out | Missing employer unit economics and moat outweigh observed value of activating existing employee benefits. |
| mac-conwell | 130-franshares-investing-in-franchises | In → Out | Generic price-and-economics caution misses investor-specific acceptable price and alternative-assets experience. |
| mac-conwell | 133-podcast-ai-josh-gets-displaced | Out → In | Early revenue and sales conversion outweigh investor experience of replication, competition and capital needed to win. |
| michael-hyatt | 39-this-pitch-is-damn-near-perfect | In → Out | Valuation/transaction scrutiny outweighs founder appeal; actual investment was conditional on family-office product testing. |
| michael-hyatt | 41-can-this-startup-help-retailers-take-on-amazon | In → Out | Traction/mature economics outweigh observed founder workability, merchant demand and shipping-access thesis. |
| michael-hyatt | 43-get-this-party-startup-in-here | Out → In | Early recurring revenue outweighs observed model discomfort and capital required for scaling. |
| michael-hyatt | 51-this-bot-can-fight-your-atm-fees | In → Out | Consumer/enterprise preference and missing economics outweigh product appeal; actual interest conditional on lead and market understanding. |
| mike-ma-sidecut-ventures | 185-its-electric-beating-tesla-was-the-easy-part | Out → In | Climate traction and founder case outweigh capital duration, dilution and valuation relative to stage. |
| mike-ma-sidecut-ventures | 187-paysquad-buy-now-pay-together | Out → In | Founder/payments experience and traction outweigh actual investor experience about weak exclusivity and competition. |
| mike-ma-sidecut-ventures | 191-haiven-ai-protection-for-young-gamers | Out → In | Founder exception overrides observed lack of action and coachability; founder credentials are not demonstrated responsiveness. |
| paige-doherty-behind-genius-ventures | 115-gemist-the-crown-jewel-of-venture | Out → In | Revenue and vertical-software fit outweigh dilution, founder incentives and capital needed against incumbent. |
| paige-doherty-behind-genius-ventures | 117-lotus-hardware-hail-mary | In → Out | CONFIRMED input/outcome mismatch: input ends at $14m cap; final in-room In was negotiated at $6m after temporary Out at $8m. No positive precedent available after holdout. |
| paige-doherty-behind-genius-ventures | 126-handle-the-uncut-pitch | Out → In | Commitment/vehicle boundary: actual says interested in a small personal check and happy to chat, after fund valuation objection; model promotes interest to concrete In. |
| rohit-gupta-future-communities-capital | 186-supercool-earth-making-it-rain-literally | In → Out | Prior relationship/information: actual says missed last round and wants this one; cold-pitch model applies technical/commercial proof bar. |
| sarah-downey-accomplice | 62-pivot-or-die | In → Out | Pivot/missing economics outweigh founder vision and timing. Available held-out corpus contains four Outs and no Ins. |
| sarah-downey-accomplice | 82-a-deal-too-perfect-to-pass-up | In → Out | Venture-return criteria oppose observed founder/angel investment. Original rationale also includes post-founder debrief; later reversal must stay separate. |
| sarah-downey-accomplice | 87-uber-for-pets | In → Out | Net economics and defensibility outweigh founder fieldwork and personal pet affinity; later Out is not the in-room label. |
| sheel-mohnot-better-tomorrow-ventures | 79-the-anti-juul | In → Out | Regulatory/capital/scale requirements dominate observed mission-driven personal-check offer. |

## Per-Investor Results

| Investor | Correct | Completed |
|---|---:|---:|
| beck-bamberger-bad-ideas-group | 4 | 5 |
| charles-hudson-precursor-ventures | 1 | 5 |
| cyan-banister-long-journey-ventures | 1 | 4 |
| elizabeth-yin-hustle-fund | 1 | 5 |
| howie-diamond-pure-ventures | 4 | 6 |
| jake-chapman-marque-ventures | 4 | 5 |
| jenny-fielding-everywhere-ventures | 4 | 4 |
| jesse-middleton-flybridge | 1 | 5 |
| jillian-manus-structure-capital | 2 | 4 |
| mac-conwell | 3 | 7 |
| mark-phillips-11-tribes-ventures | 4 | 4 |
| michael-hyatt | 4 | 8 |
| mike-ma-sidecut-ventures | 2 | 5 |
| nicole-verkindt | 4 | 4 |
| paige-doherty-behind-genius-ventures | 1 | 4 |
| phil-nadel-forefront-venture-partners | 4 | 4 |
| rohit-gupta-future-communities-capital | 4 | 5 |
| sarah-downey-accomplice | 3 | 6 |
| sheel-mohnot-better-tomorrow-ventures | 3 | 4 |
