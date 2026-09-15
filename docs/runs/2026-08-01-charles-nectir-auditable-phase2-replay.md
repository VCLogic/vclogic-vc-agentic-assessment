# Charles Hudson / Nectir: Auditable Phase 2 Replay

Machine-readable diagnostic: `2026-08-01-charles-nectir-auditable-phase2-replay.json`.

## Outcome

The replay preserved the original frozen Phase 1 investigation and ran one new
Investment Decision Synthesis call with `openai/gpt-5.6-luna`. The output was
valid on the first attempt:

| Field | Result |
|---|---:|
| Evaluated actual label | In |
| Prediction | Out |
| Investment likelihood | 0.38 |
| Decision confidence | 0.78 |
| Ranking score | 0.62 |
| Check tier | `no_check_tier` |
| Phase 2 input tokens | 4,455 |
| Phase 2 output tokens | 3,064 |
| Provider-reported cost | $0.0023952 |
| Elapsed model time | 23.12 seconds |

The result remains a false negative. The revised schema improves observability,
not accuracy on this case.

The audited label file deliberately retains more than one endpoint: Charles's
pitch-window response was conditional and recorded as Out, while the evaluated
final decision is In after the portfolio-conflict condition was resolved. The
final label is the target used here, in accordance with the revised evaluation
decision. This distinction matters: a pitch-only system cannot observe the
later conflict-check result.

## What the LangGraph process actually did

### Phase 1: frozen rationale investigation

Phase 1 ran two passes. In each pass, the planning model exhausted its 4,096
output-token allowance and returned no parseable query plan. The graph therefore
used its generic repair plan both times:

- questions: strongest positive signal and what could make the company fail;
- searches: `founder team` and `market constraints`.

Both passes consequently searched the same terms and retained the same ten
broad wiki chunks:

1. two `founder_team` chunks;
2. the Charles decision policy;
3. people-first selection;
4. fund mandate;
5. portfolio and constraints;
6. operating constraints;
7. investor-fit constraints;
8. market opportunity; and
9. competition and defensibility.

This access set is relevant to the pitch, but the process was not truly adaptive:
it did not issue targeted follow-up searches about education procurement,
institutional adoption, retention, pricing, or exception behavior. The second
pass reread the same evidence. The investigation model nevertheless produced a
comprehensive set of 13 rationales because the retrieved chunks themselves were
large and broad.

The frozen rationale inventory was:

| ID | Rationale | Direction | Salience | Confidence |
|---|---|---|---|---:|
| R1 | founder motivation/authenticity | Positive | Primary | 0.97 |
| R2 | founder execution | Positive | Primary | 0.94 |
| R3 | founder skillset | Positive | Secondary | 0.86 |
| R4 | market-problem validation | Positive | Primary | 0.91 |
| R5 | product differentiation | Positive | Primary | 0.87 |
| R6 | product adoption | Positive | Secondary | 0.90 |
| R7 | traction repeatability | Negative | Primary | 0.86 |
| R8 | growth-projection risk | Negative | Primary | 0.92 |
| R9 | venture-scale concern | Negative | Primary | 0.80 |
| R10 | market timing | Negative | Secondary | 0.89 |
| R11 | fundraising demand | Positive | Secondary | 0.90 |
| R12 | stage/valuation mismatch | Negative | Primary | 0.95 |
| R13 | dilution concern | Negative | Secondary | 0.88 |

The investigation is substantively plausible: it recognizes strong founder
execution, paid institutional validation, adoption, and distribution while also
capturing enterprise-sales, scale, timing, and financing risks. Its weakness is
not absence of rationales; it is the generic, repetitive retrieval path and a
tendency to turn absent pitch details into multiple high-confidence negatives.

### Phase 2: observable decision synthesis

The replay assessed all 13 frozen rationales exactly once, then recorded six
continuous likelihood movements:

| Step | Synthesis question | Change | Result |
|---|---|---:|---:|
| D1 | Is the problem real and can the founder execute the initial wedge? | 0.50 → 0.65 | Above threshold |
| D2 | Does LMS integration create a durable distribution advantage? | 0.65 → 0.67 | Small increase |
| D3 | Can traction convert predictably into the forecast? | 0.67 → 0.45 | Large decrease |
| D4 | Is there a demonstrated venture-scale path? | 0.45 → 0.40 | Decrease |
| D5 | Does the strongest positive countercase overcome those risks? | 0.40 → 0.43 | Small increase |
| D6 | Do financing and fund-fit issues justify passing? | 0.43 → 0.38 | Final Out |

The controlling rationales were R4, R7, R8, R9, R10, R12, and R13. The model
gave decisive positive weight to paid institutional validation (R4), but treated
sales repeatability, forecast credibility, venture scale, and round fit as
decisive negatives. It explicitly avoided double-counting correlated adoption
signals. Its strongest counterargument correctly described Nectir's five paid
campus contracts, renewal, student adoption, LMS embedding, and reported
learning improvement.

The reasoning is internally coherent, but it answers a strict standard-check
question. It treats missing mature metrics as reasons to pass and makes the
$3 million round/fund-mandate mismatch decisive. That is a poor match for the
observed Charles outcome, which was willingness to participate with a $50,000–
$100,000 check subject to a conflict check.

## Difference from the original LangGraph result

The original Phase 2 output was Out at 0.36 likelihood, 0.84 confidence, and
0.72 ranking score. It named almost the same controlling rationales but did not
show how they changed the likelihood. The auditable replay is Out at 0.38.

The new trace therefore demonstrates stability rather than correction: Luna
again saw strong positives, then allowed repeatability, scale, and fund-fit
concerns to dominate. The benefit is diagnostic clarity. We can now locate the
threshold crossing exactly at D3 and see that the countercase at D5 recovers
only 0.03.

## Difference from the Codex-based attempt

The Codex pipeline produced two explicit endpoints:

- **any check:** In at 0.61, recommending `exploratory_lt_100k`;
- **standard check:** Out at 0.22.

Its rationale set substantially overlapped with LangGraph's. Both found strong
founder execution, motivation, problem validation, and adoption; both found
negative repeatability, venture-scale, and valuation signals. The difference
was not primarily rationale discovery. It was decision semantics.

The Codex Phase 2 separated “would Charles invest any fund capital?” from “would
Charles make a standard check?” and applied an optionality gate. Paid deployment,
multi-year onboarding, LMS-origin usage, and a price-increasing renewal were
enough for a bounded exploratory check, while missing mature evidence limited
check size rather than forcing Out. That endpoint matches Charles's observed
$50,000–$100,000 interest. The LangGraph binary endpoint collapsed those two
questions and behaved like the stricter standard-check head.

There was also a tooling difference. The Codex agent consulted named wiki files
directly and recorded file-level evidence. The original LangGraph Phase 1 used
embedding retrieval over chunks; in this episode its planner failed, so generic
fallback queries selected broad sections. The new replay deliberately made no
wiki access at all because it tested only Phase 2 from the frozen investigation.

## Assessment and next implication

The rationale content is generally sensible and evidence-linked. The main
failure is downstream weighting and endpoint definition, compounded by a Phase
1 planner failure that the workflow currently hides behind a generic fallback.
The single binary decision asks one number to represent two materially different
actions. Nectir is the clearest example: both systems agree it does not clear a
standard-check bar, while only the dual-check system can express Charles's
observed small-check willingness.

The next controlled experiment should preserve this frozen Phase 1 and compare
the single binary endpoint with an explicit any-check/standard-check endpoint.
Separately, Phase 1 should expose planner fallback as a validation event and use
a concise planner output budget so that a repeated generic plan cannot be
mistaken for adaptive wiki investigation.

## Artifact locations

- Frozen source Phase 1: `outputs/openrouter-luna-charles-comparison/127-nectir-the-classroom-of-the-future/phase1/`
- Replay provenance: `outputs/openrouter-luna-charles-deliberation/127-nectir-the-classroom-of-the-future/phase1-provenance.json`
- Auditable decision: `outputs/openrouter-luna-charles-deliberation/127-nectir-the-classroom-of-the-future/phase2/decision.json`
- Complete model call record: `outputs/openrouter-luna-charles-deliberation/127-nectir-the-classroom-of-the-future/phase2/turn-01/decision-model-response.json`
- Frozen replay state: `outputs/openrouter-luna-charles-deliberation/127-nectir-the-classroom-of-the-future/state.json`
