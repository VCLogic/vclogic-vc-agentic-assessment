# Error Analysis: New Cases With Assessment Repairs

## Scope and conclusion

Reviewed all 21 saved assessments, all 11 wrong decision justifications, their final investigations, explicit decision-linked reference rationales, and the corresponding raw transcript turns. Compared correct cases and inspected shared decision instructions, policy checks, historical coverage, provider responses, and stopping logic. No paid reruns, changes to assessment behavior, or changes to original results were made.

The central problem is poor prediction of **which considerations control this particular investor's commitment**. The system frequently produces a plausible generic investment assessment. It sometimes misses the actual consideration, sometimes reverses its direction, and sometimes identifies it but discounts it. More history and structurally valid policy comparisons did not solve that judgment problem.

Supporting case-level extracts and comparisons are in [repaired-v3-error-analysis.json](repaired-v3-error-analysis.json). The original [batch progress/results](all-vcs-one-in-one-out-repaired-v3-progress.json) remain unchanged.

## Score and interpretation

| Measure | Result |
|---|---:|
| Completed executions | 21/21 |
| Decision matches | 10/21 = 47.6% |
| Actual In correctly predicted | 3/6 = 50.0% |
| Actual Out correctly predicted | 7/15 = 46.7% |
| False In / false Out | 8 / 3 |
| Accepted decisions correct | 4/7 |
| Provisional decisions correct | 6/14 |
| Always-Out reference baseline | 15/21 = 71.4% |
| Balanced accuracy | 48.3% |
| Cost | $2.49315804 |

The always-Out comparison exposes the class imbalance; it is not a useful investment policy. This small selected batch does not establish a population accuracy or a causal regression from the earlier batch, which used different cases. The four-case repair regression was selected from known failures and did not demonstrate generalization.

## All 11 wrong decisions

The actual reasons below were checked against raw source turns. Target VC remarks are held-out evaluation evidence, not material that should be inserted into that case's assessment input.

| VC / pitch | Actual → predicted | What the investor said | What the model did | Failure location |
|---|---|---|---|---|
| Beck / RedDrop #129 | Out → In | Crowded category and outside her usual scope (turn 125). | Favored revenue, margins, social benefit and ambition; acknowledged concerns but found no hard veto. Compared Outlines/ChurchSpace Ins with RobeCurls Out. | Phase 1 omitted both exact decisive taxonomy labels; Phase 2 treated category concerns as insufficient to reject. |
| Charles / Wonderade #189 | Out → In | The marketing did not resolve parental beliefs about juice or whether parents versus children drive the brand (237). | Recognized incomplete consumer/retail proof, then invoked founder experience and a small-check exception. | Marketing rationale was neutral instead of negative; consumer-belief inertia missing. Exception compared broad CPG execution risks rather than the specific positioning problem. |
| Elizabeth / PaySquad #187 | Out → In | Thin obtainable transaction economics and difficulty reaching enough large merchants (139). | Treated merchant integrations, reported demand and founder experience as sufficient; scale uncertainty became diligence. | Go-to-market and problem validation were positive instead of negative. Venture scale neutral. Concrete merchant-volume × adoption × net-take concerns did not control. |
| Howie / Skylights #4 | Out → In | Too early to invest (107). | Correctly identified missing paid repeatability, then used an aerospace investment as an exception for specialized founders before commercialization. | Rationale detected with correct negative direction, but overridden by a broad analogy. |
| Jesse / PaySquad #187 | Out → In | Price compression, tiny net take, copying/bypass risk, and insufficient maturity (144, 148). | Favored founder relevance, integrations and expansion possibilities; forecast and economics concerns lowered confidence. | Business-model/viability and repeatability labels missing; differentiation neutral. Generic venture potential replaced the specific revenue-capture test. |
| Mac / RedDrop #129 | In → Out | Strong product/company enthusiasm, concrete $50k commitment, and first-institutional-check identity (120, 138). | Put unresolved retention and a missing $100m growth bridge ahead of traction and product appeal; favored a CPG Out analogy. | Actual explicit positive labels absent; stricter growth-proof bar than the observed commitment. |
| Mac / ChurchSpace #128 | Out → In | Explicitly tends to stay away from this space (213). | Treated it as an attractive marketplace with founder fit; no hard constraint. | Target-specific sector aversion not represented in final rationale labels. Focused wiki search did not find an obvious church/faith aversion; independent memory coverage may be insufficient. |
| Michael / Shipsi #41 | In → Out | Important shipping problem, positive answers, humility and ability to work together (139). | Emphasized immature revenue, implementation, economics and pricing. | Problem validation detected, but coachability/communication and product-success labels missing. The real investor accepted open diligence. |
| Nicole / apparel pitch #39 | Out → In | Interested in founder/product but could not commit because apparel was unfamiliar and saturated (106). | Explicitly knew her clothing aversion, yet selected a founder exception and small check. | Competition and domain-knowledge labels missing. Exception comparison itself admits its historical In does not establish tolerance for apparel aversion. |
| Paige / Handle #126 | Out → In | Businesses previously backed at that valuation had much higher revenue (301). | Recognized $15m pre-money, expansion and pricing concerns, but favored operating metrics and the Lotus In analogy; no hard veto. | Negative stage/valuation mismatch detected and cited as controlling, but not given enough weight in final aggregation. |
| Sarah / SpotOn #87 | In → Out | Personal pet affinity, founder hands-on execution, and perceived large potential justified $25k (175). | Favored thin platform revenue, scaling costs and Uber competition. | Founder execution positive but not controlling; mission affinity and market-size label absent. Sarah later passed, but the requested target is her in-room In, so Out remains wrong. |

## What failed across cases

### 1. Soft preferences are being weakened into non-reasons

The policy instructions correctly distinguish soft preferences from hard prohibitions. However, several outputs then reason that, because no hard veto is established, a small investment is justified. That conclusion does not follow. A strong preference, valuation discomfort, insufficient conviction, or opportunity cost can explain an Out without a categorical rule.

Beck/RedDrop and Paige/Handle explicitly demonstrate this. Nicole is stronger evidence: relevant negative information existed and was acknowledged, but lost in the final tradeoff. Mac/ChurchSpace may additionally reflect missing independent memory; it should not be "fixed" by inserting the held-out refusal into Mac's input for that case.

### 2. The exception route passes form checks without proving the analogy

Four decisions used `founder_conviction_exception`: only Rohit/Supercool was correct; Charles, Howie and Nicole were wrong. Seven of eight false Ins used the exploratory check tier; Paige's structured tier was `standard` despite exploratory language in its justification.

Nicole's comparison says the historical investment concerned a marketplace/hair business and **does not prove** tolerance of clothing aversion. Nevertheless, the model makes the exception. Howie's analogy spans aerospace and airline entertainment; it compares pre-commercial complexity, but does not establish equal tolerance of the relevant paid-demand gap. Charles's comparison does not resolve his actual positioning concern.

The validator in `src/vc_clone_graph/policy.py` checks evidence IDs, an observed In, and the four comparison sections. It does not validate the semantic conclusion that the cited precedent tolerates the same risk. These outputs satisfy the form while leaving the central inference unsupported.

### 3. The economics instruction is not consistently applied

PaySquad is a useful two-investor example. Both predictions overvalue merchant access, completion and expansion language while underweighting how much revenue the business can retain. The model does not entirely miss the uncertainty: it mentions GMV versus revenue and unproven forecasts. It fails to make the revenue-capture question decisive.

Conversely, Mac/RedDrop, Michael/Shipsi and Sarah/SpotOn are rejected for incomplete growth/economics proof despite actual pitch-stage commitments. We therefore cannot solve this by globally increasing conservatism. The relevant bar must be investor-specific and supported by independent examples.

### 4. Rationale detection is substantially worse than decision accuracy suggests

Using the existing automated evaluator for exactly these 21 job IDs:

| Exact taxonomy metric, micro-averaged | Precision | Recall | F1 |
|---|---:|---:|---:|
| Label only | 36.3% | 49.0% | 41.7% |
| Label plus direction | 17.6% | 23.8% | 20.2% |

Counts: label-only TP 101 / FP 177 / FN 105; label+direction TP 49 / FP 229 / FN 157. These are diagnostic comparisons against automated references, not human-certified scores. Synonyms, related labels, sparse observed speech and reference timing errors limit their interpretation.

Examples separate failure types:

- **Omission:** Beck's competition/thesis reasons; Michael's coachability/communication.
- **Wrong direction:** Elizabeth's PaySquad go-to-market is positive in the model versus negative in observed remarks.
- **Detected but outweighed:** Howie's early-stage concern and Paige's valuation mismatch.
- **Right decision, incomplete rationale:** Jenny/CurieDx is correctly Out, but the model makes founder communication positive against a negative reference. Mike/Haiven is correctly Out while founder execution/coachability are positive against negative references.

A decision match alone is insufficient evidence that the VC has been modeled correctly.

### 5. Historical coverage is still uneven

Accessible labeled history varies substantially after target exclusion: Nicole has one In and one Out (seven accessible records including other statuses); Paige one In/six Out; Sarah/SpotOn one In/eight Out; Jenny zero In/two Out. Charles has 15 In/58 Out, so sparse memory is not an explanation for every mistake.

More history does not automatically mean better comparisons. Broad resemblance can select persuasive but weak analogies, and a single positive example can be overgeneralized. This inspection did not establish that the closest valid precedent was absent from the index versus present but poorly retrieved in every case; a retrieval-ranking audit would be needed for that distinction.

### 6. Accepted status is not correctness

Accepted means the workflow finalized the candidate under its sufficiency/guardrail rules. It does not mean independently verified factual or predictive correctness. Three of seven accepted decisions were wrong.

Seven runs ended at the iteration limit and seven after repeated corpus searches. That explains provisional status, but it does not itself explain which predictions are wrong. Repeating the same searches is unlikely to repair a bad preference tradeoff.

### 7. Structured-output problems remain a secondary burden

Across 271 saved phase-1/phase-2 response artifacts, 11 had no parsed output and 11 hit the output limit; 80 used JSON-object fallback. Twenty cases recorded an invalid plan attempt, and nine used a planning fallback. Some invalid plans are valid JSON that fail schema bounds rather than parse errors. Saved null-parsed response costs total about $0.072; this is not the total cost of every invalid candidate or hidden physical attempt.

All 21 ultimately produced completed results. The substantive wrong decisions are visible in usable final artifacts, so generation failures alone do not explain the score. We have not run an ablation to measure how much fallback planning harmed retrieval quality.

## Evaluation-data issue discovered

PaySquad's founder thanks the panel at turn 153, followed by the panel's goodbye/applause at 156. The post-founder discussion starts at 157. Yet several reference entries are tagged `original_pitch`: Elizabeth's turns 197, 255, 257, 259, 263 and 265, and Jesse's 227, among the explicit-reason comparisons examined here.

Those observations may explain the decision retrospectively, but they should be separated from live-pitch rationales if that is the intended scoring target. The pre-exit remarks at Elizabeth 139 and Jesse 144/148 independently support the economic concerns, so their wrong decision classifications do not depend on the mislabeled later remarks.

References are machine extracted and independently model reviewed, not human validated. Original annotations and scores were preserved; this report flags the issue rather than silently changing ground truth.

## Recommended next work, in order

1. **Repair and audit reference timing**, separating live-pitch reasoning from post-founder explanation. Manually review this 21-case set's decisive rationale labels/directions before using them as a trusted benchmark.
2. **Evaluate the stages separately:** missing decisive consideration, wrong direction, and wrong final weight. Do not optimize only the final In/Out score.
3. **Require positive support for an In.** Absence of a hard veto must not substitute for evidence that this investor would commit. Preserve soft negative preferences as decision-relevant without promoting them to absolute rules.
4. **Validate exception meaning**, including whether an acknowledged difference defeats the same-risk analogy. A valid H-ID and four headings are not enough. Compare both In and Out examples on the specific concern.
5. **Make economic comparisons explicit where supported:** revenue captured, obtainable paying customers, adoption, take rate and distribution costs. Treat missing evidence according to the investor's historical bar, rather than applying one universal seed-stage standard.
6. **Measure changes on untouched cases.** These 21 cases are now diagnostic/development examples; reusing them can validate behavior changes but cannot establish fresh generalization. Keep repeated runs separate and preserve earlier results.

These recommendations are based on observed failure mechanisms, not a causal proof that any one change will improve accuracy. No further implementation or paid run was performed as part of this analysis.

## Leakage qualification

This investigation does not replace a full adversarial leakage audit. Known wiki source-lineage and target-inclusive aggregate gaps remain. Sarah's prediction matching her later Out is not, by itself, proof of leakage: the cleaned founder input itself discusses Uber competition. Future references remain allowed when they do not disclose the target's decision or rationale, consistent with the project rule. No leakage-clean accuracy claim is made here.
