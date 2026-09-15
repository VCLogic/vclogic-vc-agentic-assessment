# Charles Hudson: LangGraph Luna versus Codex comparison

## Executive result

The LangGraph/OpenRouter implementation ran successfully on all three selected Charles
Hudson episodes. Every Phase 1 rationale investigation and Phase 2 decision artifact was
accepted and hash-verified, with no retries.

As a binary classifier, LangGraph Luna predicted all three pitches `Out`. It therefore
classified one of three revised labels correctly: Harper Wilde (`Out`). It missed Shipsi
and Nectir, whose project labels are revised final `In` decisions. The selected Codex
baselines also classified one of three correctly, but on a different episode: Codex
recovered Nectir through its specialized any-check endpoint while missing Shipsi and
Harper Wilde.

As a prioritization system, LangGraph performed much better in this deliberately small
comparison. It ranked Nectir first, Shipsi second, and Harper Wilde third. The two actual
Ins were therefore the top two pitches: precision@2 and recall@2 were both 100%. This is
illustrative evidence, not a performance estimate; the sample was deliberately selected
and contains only three cases.

## Experimental design

The LangGraph runs used:

- `openai/gpt-5.6-luna` through OpenRouter for both phases;
- local Ollama `nomic-embed-text` embeddings for hybrid Charles-wiki retrieval;
- the same byte-identical `a.txt` pitch cuts used by the selected Codex runs;
- the same Charles investment wiki and 44-label rationale taxonomy;
- at most two Phase 1 passes and two Phase 2 passes;
- no actual labels, full transcripts, previous predictions, or evaluation reports inside
  the inference package.

Phase 1 searched the Charles wiki iteratively and produced evidence-bound rationales but
no verdict. Phase 2 received only the frozen Phase 1 investigation and synthesized the
decision, likelihood, confidence, check tier, and independent ranking score.

The Codex comparison is not a pure orchestration-only ablation. Episodes 20 and 41 used
Luna for both Codex phases. Nectir used Sol for Codex Phase 1 and Luna with a specialized
dual-check schema for Phase 2. The new LangGraph runs used Luna throughout and one generic
decision schema. Differences can therefore arise from orchestration, schema, retrieval,
prompting, and—on Nectir—the Phase 1 model.

## Label interpretation

The project’s revised final labels are used, consistent with the user-approved audit:

| Episode | Revised final label | Decision context |
|---|---:|---|
| Harper Wilde | Out | Charles declined during the initial panel decision. |
| Shipsi | In | Charles expressed conditional interest on-panel; conflict resolution occurred later. |
| Nectir | In | Charles offered $50K–$100K conditional on no portfolio conflict; the condition was later treated as resolved. |

This distinction materially affects interpretation. The pitch-window audit records Shipsi
and Nectir as conditional rather than unconditional current commitments. A pitch-only
system does not observe the later conflict-resolution information embedded in their final
labels. The binary misses are real under the revised-label evaluation, but they are not
equivalent to missing two unconditional pitch-window checks.

## Outcome comparison

| Episode | Actual | LangGraph decision | Likelihood | Ranking | Codex endpoint | Codex likelihood |
|---|---:|---:|---:|---:|---:|---:|
| Nectir | In | Out | 36% | **0.72** | Any-check In; standard-check Out | 61%; 22% |
| Shipsi | In | Out | 32% | **0.66** | Out | 32% |
| Harper Wilde | Out | **Out** | 38% | **0.58** | In | 58% |

Binary correctness was 1/3 for each approach on this set. LangGraph had zero In recall
and perfect Out recall; the selected Codex endpoints had 50% In recall and zero Out
recall. These rates should not be generalized from three purposefully selected examples.

## Rationale comparison

Across the three cases, the mean Jaccard overlap between LangGraph and Codex rationale
label sets was 41.7%. Among the 21 shared labels, 15 had the same direction, for 71.4%
directional agreement. This indicates meaningful common structure despite substantial
differences in what each process chose to make explicit.

### Harper Wilde

- Codex: 12 rationales; LangGraph: 8.
- Shared labels: 5 of 15 in the union; Jaccard overlap 33.3%.
- Direction agreement: founder execution, founder motivation authenticity, and market
  problem validation were positive in both.
- Direction disagreement: Codex treated differentiation and venture scale as negative;
  LangGraph preserved them as unresolved/neutral.
- LangGraph uniquely made unit economics negative and included product adoption and GTM
  assessment. Codex uniquely represented founder-market fit, founder communication,
  competitive inertia, product success factors, market entry, repeatability, and a general
  business-model assessment.

Both systems recognized strong founder and problem signals. Codex allowed those signals
to outweigh category and economic concerns, producing the wrong `In`. LangGraph required
paid conversion and contribution-margin evidence and produced the correct `Out`. However,
Charles’s actual reason was a mismatch in brand/opportunity vision. Because that investor
reaction was excluded from the pitch-only input, LangGraph’s correct label should not be
described as exact recovery of Charles’s observed reasoning.

### Shipsi

- Codex: 13 rationales; LangGraph: 12.
- Shared labels: 7 of 18 in the union; Jaccard overlap 38.9%.
- Direction agreement: both made founder execution, problem validation, and solution
  feasibility positive; growth projection risk negative; and unit economics unresolved.
- Direction disagreement: LangGraph treated product differentiation as positive rather
  than negative and repeatability as neutral rather than negative.
- LangGraph added explicit competitive inertia, deal complexity, GTM, market-size, and
  venture-scale assessments. Codex separately represented adoption, business model,
  market entry, timing, execution risk, and stage/valuation fit.

Despite giving Shipsi more credit for differentiation and repeatability than Codex, the
LangGraph synthesis still returned `Out`. It focused on pre-revenue status, unlaunched
implementations, founder-dependent support, uncertain margins, aggressive projections,
and platform disintermediation. Both systems missed the revised final `In`. The actual
decision was conditional on resolving portfolio conflicts, which the inference system
could neither inspect nor resolve from the pitch.

### Nectir

- Codex: 13 rationales; LangGraph: 13.
- Shared labels: 9 of 17 in the union; Jaccard overlap 52.9%, the highest of the three.
- Direction agreement: both made founder execution, motivation, problem validation, and
  product adoption positive; valuation mismatch, repeatability concern, and venture-scale
  concern negative.
- Direction disagreement: Codex treated timing as neutral while LangGraph made it negative;
  Codex made differentiation negative while LangGraph made it positive.
- LangGraph added founder skillset, fundraising demand, dilution, and projection risk.
  Codex separately represented founder-market fit, founder communication, GTM, and the
  business model.

The substantive assessments were similar: real paid institutional adoption and strong
founder execution coexisted with valuation, sales-repeatability, market-scale, and financing
concerns. The key difference occurred during decision synthesis. The specialized Codex
dual-check process converted those positives into a 61% `In` for a sub-$100K exploratory
check while retaining a 22% `Out` for a standard check. The generic LangGraph synthesis
returned `Out`, `no_check_tier`, and 36% likelihood, although it ranked Nectir highest at
0.72. This is evidence that the optionality gate—not rationale discovery—is the main lost
mechanism in this case.

## What the comparison suggests

First, replacing Codex CLI orchestration with LangGraph did not by itself improve binary
classification on these cases. Both approaches achieved one correct outcome, and the
LangGraph endpoint remained conservative.

Second, LangGraph preserved useful prioritization information. Its ranking placed both
revised Ins above the Out even though neither crossed the 0.50 investment threshold. This
supports retaining separate classification and diamond-search evaluation.

Third, the Nectir comparison isolates an actionable design difference: the earlier Codex
method explicitly separated “any check” from “standard check,” while LangGraph currently
asks for one decision plus a check-tier field. A single synthesis threshold is not
equivalent to a two-gate capital-allocation decision. The current run should be frozen as
the generic-schema baseline; a future preregistered comparison can test a LangGraph
dual-check schema without rerunning Phase 1.

Fourth, correctness and process fidelity must remain separate. Harper Wilde was classified
correctly for economic and category reasons, not for Charles’s observed brand-vision
reason. Shipsi and Nectir were classified incorrectly under final labels partly because
their labels incorporate post-pitch conflict resolution. Future reporting should present
both the final investment label and pitch-window decision context.

## Usage and cost

| Episode | Phase 1 passes | Phase 2 passes | Input tokens | Output tokens | OpenRouter cost |
|---|---:|---:|---:|---:|---:|
| Shipsi | 1 | 1 | 28,745 | 4,757 | $0.00644710 |
| Nectir | 2 | 1 | 61,958 | 14,907 | $0.01668858 |
| Harper Wilde | 2 | 1 | 64,061 | 6,532 | $0.01188475 |
| **Total** | **5** | **3** | **154,764** | **26,196** | **$0.03502043** |

OpenRouter reported no cached input tokens. Local Charles-wiki indexing used Ollama and
incurred no OpenRouter cost.

## Reproducibility and artifacts

Machine-readable comparison data is in
`docs/runs/2026-08-01-charles-langgraph-vs-codex.json`. The three configurations are:

- `configs/openrouter-luna-charles-amazon.toml`
- `configs/openrouter-luna-charles-nectir.toml`
- `configs/openrouter-luna-charles-harper-wilde.toml`

Frozen LangGraph artifacts are under `outputs/openrouter-luna-charles-comparison/`. The
Codex provenance paths, complete rationale maps, per-case overlap calculations, usage,
and endpoints are recorded in the machine-readable report.
