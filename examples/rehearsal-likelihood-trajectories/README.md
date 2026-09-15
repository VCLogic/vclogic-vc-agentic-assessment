# Rehearsal likelihood-change examples

Generated on 2026-09-01 with the live LangGraph v4/v4.1-grounded rehearsal workflow and the fictional ShiftPilot pitch in `demo-pitches/shiftpilot-yin-rehearsal.md`.

The founder answers below are synthetic demonstration evidence. They are intentionally different across the two sessions so the examples show how genuinely new positive and negative evidence changes an investor-like assessment. They are not claims about a real company.

## In example: Elizabeth Yin-like investor

- Session: `a2d9811c-9735-41d3-b4bf-f4e7ccbc13f4`
- Web route: `/sessions/a2d9811c-9735-41d3-b4bf-f4e7ccbc13f4`
- Artifact root: `outputs/rehearsals-v41-grounded/elizabeth-yin-hustle-fund/a2d9811c-9735-41d3-b4bf-f4e7ccbc13f4/`
- Initial assessment: **In, 61% likelihood, 64% confidence**
- Answer-level trajectory: **61% → 68% → 74% → 79%**
- Final synthesis: **In, 71% likelihood, 74% confidence**

| Turn | New evidence | Net effect | Likelihood | Activated or updated rationales |
|---|---|---:|---:|---|
| 1 | Five customers came through one repeatable association channel; deployment no longer required founders; the cohort produced $12,100 MRR with no observed churn over six months. | Positive | 61% → 68% | Product adoption; founder execution; traction repeatability; go-to-market strategy; solution feasibility |
| 2 | Fully loaded acquisition and implementation cost was $3,900; contribution-margin payback was 2.1 months; post-onboarding gross margin was approximately 83%. | Positive | 68% → 74% | Unit economics; go-to-market strategy |
| 3 | Embedded configurations, integrations, and decision histories reduced median replacement time; four customers routed all replacements through the product. | Positive | 74% → 79% | Competitive advantage; product adoption |
| Final | Phase 2 rebalanced the new evidence against remaining cohort, financing, ROI, incumbent, and expansion risks. | Synthesis | 79% → 71% | Final decision synthesis across the complete rationale state |

## Out example: Charles Hudson-like investor

- Session: `5ffd241f-2d30-4f37-b71b-10c67788b7b3`
- Web route: `/sessions/5ffd241f-2d30-4f37-b71b-10c67788b7b3`
- Artifact root: `outputs/rehearsals-v41-grounded/charles-hudson-precursor-ventures/5ffd241f-2d30-4f37-b71b-10c67788b7b3/`
- Initial assessment: **Out, 27% likelihood, 81% confidence**
- Answer-level trajectory: **27% → 21% → 14% → 10%**
- Final synthesis: **Out, 10% likelihood, 87% confidence**

| Turn | New evidence | Net effect | Likelihood | Activated or updated rationales |
|---|---|---:|---:|---|
| 1 | Adjacent customer segments produced no paid pilots; only about 900 agencies appeared ready to buy at the current price; expansion remained unvalidated. | Negative | 27% → 21% | Market size; traction repeatability; investment upside; newly activated venture-scale concern |
| 2 | No customer purchased an add-on; three declined expansion pricing; one stopped using an analytics prototype. | Negative | 21% → 14% | Business model; unit economics; traction repeatability; investment upside; product adoption |
| 3 | Among three observable renewals, one renewed fully, one contracted by 30%, and one churned back to an incumbent system. | Negative | 14% → 10% | Competitive advantage; incumbent inertia; business model; traction repeatability; unit economics; investment upside; product adoption |

## What these examples demonstrate

1. A question's rationale labels are retrieval targets, not an allow-list.
2. One answer can update every taxonomy-valid rationale supported by its evidence.
3. Each changed rationale cites the exact founder answer and retains a deterministic identity.
4. Answer-level likelihood moves only for new decision-relevant evidence.
5. Final Phase 2 synthesis may recalibrate the last answer-level score after considering the complete rationale state and remaining risks.

Combined provider usage for the two demonstrations was **$0.27098625**, with **723,261 input tokens** and **75,146 output tokens** reported by the provider integration.
