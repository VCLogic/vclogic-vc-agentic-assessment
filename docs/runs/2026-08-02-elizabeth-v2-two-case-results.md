# Elizabeth Yin v2 two-case development run

## Scope

This is a two-case development check of the reusable two-phase framework using `openai/gpt-5.6-luna` through OpenRouter. The model received only the audited pitch-only transcript, Elizabeth Yin's source-linked wiki, and rationale taxonomy v2. Actual labels and complete episode transcripts remained outside the inference package and were consulted only after outputs were frozen.

The intended In case, `135-thoras-ai-the-twin-effect`, failed Phase 1 schema validation in three separately preserved attempts and therefore produced no decision. To complete the requested In/Out comparison without weakening validation, it was replaced by the audited actual-In case `131-flock-ai-model-generated-models`.

## Frozen outcomes

| Episode | Actual | Any-check prediction | Likelihood | Confidence | Ranking score | Standard-check prediction | Standard likelihood | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `131-flock-ai-model-generated-models` | In | **In** | 0.62 | 0.70 | 0.76 | Out | 0.35 | P1 provisional; P2 provisional |
| `124-ecgo-turning-trash-into-treasure-with-ai` | Out | **In** | 0.58 | 0.66 | 0.74 | Out | 0.30 | P1 accepted; P2 provisional |

For the any-check endpoint, this pair contains one true positive and one false positive. For the standard-check endpoint, it contains one false negative and one true negative. Flock ranks just above ECGo (0.76 versus 0.74), so the actual In is correctly prioritized within this two-case set, but the margin is only 0.02 and the sample is far too small for a performance estimate.

## Flock: actual In, predicted In

Phase 1 locked strong positive rationales around the concrete customer problem, founder-market fit, rapid execution, pilots and customer outcomes, product feasibility, a plausible SaaS model, early investor demand, and broad mission fit. It also captured the primary negative rationale: dependence on a commoditizing AI stack, plus valuation and repeatability concerns.

Phase 2 recommended only a `small_exploratory` check. Its logic was that pilots, LOIs, reported customer outcomes, and relevant founder experience justified preserving optionality, while a $7.5–8 million post-money valuation, uncertain defensibility, missing mature economics, and unknown allocation constrained the check size.

This aligns unusually well with Elizabeth's later, withheld episode reasoning. She praised the founder's ability to synthesize information, execute, and pivot; liked the analytics-based moat and SaaS interpretation; considered the valuation too expensive; and offered $50,000 at a $6 million post-money cap to fund a more complete customer loop. The model recovered both the positive exception logic and the valuation-based size constraint without seeing that decision.

The output is still marked provisional. Phase 1 used the planner fallback, and Phase 2's audit found that the risk ledger did not cover every expected rationale consistently. These are output-contract findings, not evidence that the frozen decision is absent.

## ECGo: actual Out, predicted In

Phase 1 identified two paid institutional deployments, user engagement, reported low acquisition cost, founder adaptation, and a university distribution wedge. Phase 2 treated these as enough to justify a small optional check despite unresolved repeatability, measurable customer outcomes, margins, deployment economics, defensibility, and deal mechanics.

The complete episode, inspected only after inference, explains the false positive. Elizabeth explicitly liked the founder and the two campus sales, but could not reconcile the B2B2C customer-acquisition economics: she questioned how adoption could expand from roughly 1,500 users to a whole campus without sharply increasing spend. She concluded that there was value in the problem but that it was not the right model. The agent recognized this uncertainty but classified it as a testable diligence gap rather than a reason to decline even a small check.

This exposes the present any-check boundary problem: the optionality rule is still too permissive when early paid pilots coexist with unresolved distribution economics. The standard-check endpoint correctly returned Out, but its stricter threshold also missed Flock's small-check In.

## Usage

| Run | Input tokens | Output tokens | Cached input | Cost (USD) |
|---|---:|---:|---:|---:|
| Flock completed run | 32,153 | 14,718 | 0 | 0.01285 |
| ECGo completed run | 32,575 | 12,808 | 0 | 0.01176 |
| Thoras failed attempt 1 | 50,411 | 15,360 | 0 | 0.01552 |
| Thoras failed attempt 2 | 54,154 | 16,973 | 2,977 | 0.01661 |
| Thoras failed attempt 3 | 50,826 | 16,222 | 2,977 | 0.01574 |
| **Total** | **220,119** | **76,081** | **5,954** | **0.07248** |

The two completed cases alone cost approximately $0.02461. Most usage in this development check came from the three Thoras failures, which are preserved for debugging rather than silently discarded.

## Interpretation

This trial supports keeping both endpoints. The standard-check decision expresses conventional underwriting discipline; the any-check decision captures Elizabeth's willingness to make a small, option-preserving investment. However, the any-check head still needs a sharper rule for distinguishing a genuinely fundable learning loop from generic unresolved diligence. The most promising discriminator from this pair is not merely the presence of pilots: it is whether the pitch demonstrates a credible, investor-relevant path to a complete and repeatable customer loop at a right-sized valuation.
