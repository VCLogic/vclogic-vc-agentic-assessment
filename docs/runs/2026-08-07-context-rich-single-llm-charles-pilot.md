# Context-rich single-LLM Charles pilot

## Purpose

This pilot implements the strongest simple-prompt baseline for comparison with the two-phase LangGraph system. One `openai/gpt-5.6-luna` request receives the audited pitch-only cut, every file in Charles Hudson's source-linked wiki, the complete rationale taxonomy, and the five most semantically relevant target-excluded historical episodes. It must directly identify rationales and return Phase 2-equivalent classification and prioritization fields in one strict JSON response.

The pilot contains the first observed In and first observed Out in the deterministic Charles round-robin order. It verifies operation and comparability; two cases are not a performance estimate.

## Execution contract

- One logical model call per pitch; no tool loop, model repair, or episode retry.
- Luna through OpenRouter, high reasoning, maximum 16,384 output tokens.
- Local pinned `nomic-ai/nomic-embed-text-v1.5` embeddings select five unique precedents.
- The current episode is physically excluded from precedent retrieval.
- Actual labels are added only after inference for ordering and scoring.
- Exact prompts, source hashes, selected precedents, raw responses, parsed results, validation findings, tokens, and costs are retained.

Both calls used one physical HTTP attempt and produced schema-valid results.

## Results

| Episode | Actual | Baseline | Any-check likelihood | Standard-check | Standard likelihood | Confidence | Ranking score | Rationales | Input/output tokens | Cost |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `39-this-pitch-is-damn-near-perfect` | In | Out | 0.20 | Out | 0.12 | 0.84 | 0.58 | 21 | 90,616 / 12,039 | $0.01855 |
| `18-rowvigor` | Out | Out | 0.05 | Out | 0.02 | 0.93 | 0.18 | 26 | 93,648 / 12,501 | $0.01921 |

Total use was 184,264 input tokens, 24,540 output tokens, and **$0.03776**. Binary classification was one correct and one incorrect. In the two-case prioritization comparison, the true In ranked above the true Out (0.58 versus 0.18), despite being classified Out.

## What the baseline reasoned

For episode 39, the model found strong positive primary rationales for founder-market fit, founder execution, communication, product adoption, and problem validation. It nevertheless placed greater decision weight on differentiation, competition, venture scale, thesis fit, and upside concerns. This is the familiar conservative-boundary error: the model recognized why the pitch was compelling and assigned it the higher review score, but treated unresolved scale and defensibility questions as sufficient to reject even a small check.

For episode 18, it identified the central negative pattern more decisively: consumer-fitness thesis friction, uncertain at-home adoption, hardware capital intensity, unclear unit economics, weak differentiation, and uncertain venture scale. Positive founder and local-community evidence remained secondary. This produced a correct Out with low investment likelihood and a low ranking score.

## Comparison with the two-phase agentic run

The previous LangGraph run did not produce a final decision for episode 39: Phase 1 failed after consuming 647,220 input tokens, 22,863 output tokens, and $0.09317. The one-call baseline produced a valid, auditable response for $0.01855, but its decision was still wrong.

For episode 18, the previous agentic run predicted a conditional exploratory **In** with any-check likelihood 0.36, standard-check Out likelihood 0.10, and ranking score 0.31. That was incorrect against the audited Out label. It used 1,606,145 input tokens, 38,878 output tokens, and $0.32703. The one-call baseline correctly predicted Out and used substantially fewer tokens and cost, while identifying 26 rationales versus 13 in the agentic Phase 1 artifact.

The comparison does not establish that the simple baseline is superior: it covers only two selected cases and missed the actual In. It does establish that simply supplying all context can generate a rich rationale record cheaply and reliably, while the separate agentic loop may consume much more context and can still either fail structurally or overuse the optional-check mechanism. The full matched baseline is therefore scientifically necessary before claiming incremental value for the agentic architecture.

## Selected precedents

Episode 39 used episodes 75, 89, 86, 73, and 18. Episode 18 used episodes 89, 69, 86, 66, and 158. All were selected from the target-filtered semantic index, and each supplied the complete transcript plus Charles's audited observed decision evidence.

## Artifact location

All run artifacts are under `outputs/context-rich-single-llm-charles-pilot-2026-08-07/`. The reproducible implementation and commands are under `baselines/context-rich-single-llm/`.
