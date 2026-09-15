# Adaptive, Leakage-Safe Charles Comparison

## Purpose

This is a four-episode diagnostic of the reusable LangGraph VC-clone workflow with
`openai/gpt-5.6-luna`. The scored target is Charles Hudson's observable binary
decision at the end of the pitch window. Later diligence outcomes are excluded.
Nectir is included as a controlled conditional-In case; Dig, Pepper, and Nectir are
Ins under the project's rule that an expressed willingness to invest subject to a
routine conflict check remains In; Harper Wilde is an explicit Out. This sample is too small and too
purposefully selected to support a general performance claim.

## What changed

The run uses one canonical Charles investment memory but derives a fresh in-memory
view for each pitch. Target-company aliases are declared in the audited package,
hash-bound by the episode manifest, removed from the memory without a visible
redaction marker, and re-embedded only for changed chunks. Canonical source hashes
are still checked before exact reads.

Phase 1 is a bounded rationale investigation. It receives the pitch-only transcript,
retrieves Charles's memory iteratively, always executes a system-owned portfolio
query, and freezes an evidence-linked rationale set without deciding. A short planner
is capped at 768 output tokens; invalid or truncated plans use a recorded deterministic
fallback. Phase 1 is capped at three passes and stops earlier on saturation or no
retrieval novelty.

Phase 2 receives only the frozen Phase 1 rationale artifact. It produces the binary
any-check decision, a stricter standard-check endpoint, likelihoods, confidence,
and review priority. Minor conditions may remain In. A material unresolved condition
normally means Out. Complete but semantically inconsistent outputs receive one
reconsideration and are then retained as provisional with quality findings.

An initial Nectir run was discarded because the literal marker
`[TARGET COMPANY REMOVED]` revealed that some holdings-list entry had been redacted.
The firewall was corrected to remove the alias cleanly, and Nectir was rerun from
Phase 1 in a fresh output and checkpoint. The result below is only the clean rerun.

## Audited labels

| Episode | Pitch-window label | Audit basis |
|---|---:|---|
| Nectir | In | Charles expressed willingness to invest $50K–$100K subject to a conflict check; under the project's minor/routine-condition rule this remains In. |
| Dig | In | “So I’m in for 10,000.” |
| Pepper | In | “So I’d like to make a small investment of $25k.” |
| Harper Wilde | Out | Charles said he and the founders did not see the opportunity the same way. |

## Results

| Episode | Actual | Any-check | Likelihood | Confidence | Standard | Std. likelihood | Review score | P1 / P2 status |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Nectir | In | Out | 0.24 | 0.94 | Out | 0.12 | 0.78 | provisional / accepted |
| Dig | In | In | 0.53 | 0.61 | Out | 0.23 | 0.70 | provisional / provisional |
| Pepper | In | Out | 0.22 | 0.82 | Out | 0.08 | 0.25 | accepted / accepted |
| Harper Wilde | Out | Out | 0.10 | 0.94 | Out | 0.05 | 0.90 | provisional / accepted |

For the requested two-In/one-Out comparison, the binary classifier is correct on
2/3 episodes (66.7%), with 50% In recall, 100% Out recall, and 75% balanced
accuracy. The likelihood ordering is more useful for screening: Dig (0.53) and
Pepper (0.22), the two actual Ins, both rank above Harper Wilde (0.10). Thus the
top-two queue has 100% precision and 100% recall in this deliberately tiny sample.

Including Nectir, the clean run is correct on 2/4 episodes (50%) with 66.7% balanced
accuracy: it recovers one of three Ins and the sole Out. Ranking is substantially
better than the fixed decisions: all three Ins rank above Harper Wilde, giving
average precision 1.0 against 0.75 positive prevalence. These figures are diagnostic only.

The separate `ranking_score` must not currently be interpreted as probability of
an In. It is a model-generated human-review-priority score, and it assigned 0.90
to Harper Wilde despite a 0.10 any-check likelihood. For diamond search, the
any-check likelihood is the coherent ranking endpoint until the priority score is
redefined and calibrated.

## Episode-level interpretation

### Nectir

The marker-free Phase 1 retrieved the mandatory portfolio evidence and retained
Campus, ClassDojo, and Wonder School while removing Nectir. It classified the
edtech adjacency as routine rather than a demonstrated direct conflict. Phase 2
nonetheless predicted Out because it treated the $3 million seed round as a
material mismatch with the retrieved sub-$1 million pre-seed construct.

The label is incorrect and the controlling explanation is contradicted by Charles's
observed behavior. Charles said he wanted to invest $50,000–$100,000 provided no
portfolio conflict existed, showing that the larger round did not itself block a
small check. Phase 1 correctly treated the portfolio overlap as routine, but Phase 2
then made the $3M round a fatal mandate mismatch. This exposes a systematic conflation
of total round size with Charles's own participation check size.

### Dig

The workflow recovered the explicit In with a bounded exploratory-check rationale.
Positive rationales included problem validation, adoption, founder authenticity,
execution, acquisition economics, and fundraising demand. Negative or unresolved
considerations included marketplace balance, incumbent competition, monetization,
venture scale, and round fit. The final decision is provisional only because the
risk ledger included contextual entries beyond the exact opposing-rationale list;
the binary decision itself was complete and stable across retries.

### Pepper

Pepper is the remaining false negative. Phase 1 recognized the specific fit problem,
purpose-built cup molds, reported CAC/LTV and margin, founder coachability, and the
second Urban Outfitters order. It then made True & Co's lingerie adjacency a
material unresolved conflict. That constraint dominated Phase 2 and produced a
high-confidence Out.

This shows that “same category” and “genuine competitive conflict” are not yet
separated reliably. The wiki names True & Co and its category but does not contain
the time-indexed, product/customer/channel evidence needed to decide whether the
overlap should be routine or veto-capable.

### Harper Wilde

The workflow corrected the historical Codex false positive. It recognized authentic
founder motivation and a resourceful 400-bra test, but also identified absent test
metrics, unclear business and inventory economics, uncertain differentiation,
unproven venture scale, and True & Co adjacency. The Out is correct, although the
system's stated reason emphasizes conflict and missing evidence rather than Charles's
actual brand-vision disagreement.

## Comparison with the earlier Codex process

| Episode | Historical Codex any-check | New LangGraph any-check | Outcome change |
|---|---:|---:|---|
| Nectir | In 0.61 | Out 0.24 | Regressed from a correct In to an incorrect Out |
| Dig | In 0.62 | In 0.53 | Correct in both |
| Pepper | Out 0.34 | Out 0.22 | Missed in both |
| Harper Wilde | In 0.62 | Out 0.10 | Corrected historical false positive |

The new and old rationale sets overlap on the core evidence but differ where the
new workflow imposes deterministic conflict retrieval and stricter source binding:

| Episode | Shared rationale themes | Important new or stronger themes |
|---|---|---|
| Nectir | founder motivation/execution, problem validation, GTM, traction repeatability, differentiation, timing, stage fit | explicit portfolio screen, founder skill gap, dilution/growth-projection decomposition |
| Dig | problem/adoption, execution, market size, competition, economics, venture scale, fund fit | explicit portfolio screen, founder authenticity, GTM, fundraising signal |
| Pepper | problem validation, differentiation, unit economics, GTM, portfolio conflict | conflict changes from neutral/secondary to negative/primary; capital intensity and competition are separated |
| Harper Wilde | problem validation, motivation, execution, differentiation, business model | product-adoption uncertainty, market-scale uncertainty, and portfolio conflict become explicit |

The historical Codex Nectir run is not a fully leakage-safe comparator because it
could inspect the canonical wiki containing the target holding. The new marker-free
run is the valid inference artifact.

## Usage

| Episode | Input tokens | Output tokens | OpenRouter cost |
|---|---:|---:|---:|
| Nectir clean rerun | 51,940 | 12,261 | $0.01349 |
| Dig, including failed strict Phase 2 and replay | 66,338 | 21,610 | $0.02126 |
| Pepper | 49,602 | 7,998 | $0.01100 |
| Harper Wilde | 39,816 | 6,414 | $0.00880 |
| Representative total | 207,696 | 48,283 | $0.05455 |

The actual live spend was $0.06537 after including the discarded visible-marker
Nectir run. Input volume remains high because Phase 1 currently supplies as many as
20 full exact wiki chunks to Luna. This is inexpensive with Luna but should be
reduced before using a more expensive production model.

## Main conclusions

1. The reusable LangGraph workflow now preserves auditable binary decisions and
   closes the direct target-name and redaction-marker leakage paths tested here.
2. The requested three-case comparison improves over the historical Codex outcomes
   on Harper and retains Dig, but it still misses Pepper. On Nectir, however, it
   regresses from a correct historical In to an incorrect Out.
3. The largest scientific and engineering gap is not rationale extraction. It is
   calibrating when Charles treats an apparent constraint as routine, material, or
   exception-worthy.
4. Portfolio memory needs time indexing and richer holding descriptions. A generic
   category label is insufficient to distinguish direct competition from harmless
   adjacency.
5. For screening, use any-check likelihood. Do not use the present review-priority
   `ranking_score` as a diamond probability.
