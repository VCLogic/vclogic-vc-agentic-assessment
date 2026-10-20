# Historical Dialogue Cleanup

The installed historical corpus contains only retained, explicitly attributed founder and panel-VC speech inside reviewed pitch boundaries, extended through the reviewed in-room decisions. This is historical memory, so its own investment decisions remain. During an assessment the target episode is still withheld; this cleanup does not change the A/B pitch inputs.

## Coverage

Processed 675 investor–episode records across 19 investors. Retained **410** and excluded **265**. Exclusions are conservative: no unverified source is substituted when cleaning cannot preserve decision evidence.

| Investor | Retained | Excluded |
|---|---:|---:|
| beck-bamberger-bad-ideas-group | 1 | 13 |
| charles-hudson-precursor-ventures | 69 | 51 |
| cyan-banister-long-journey-ventures | 15 | 13 |
| elizabeth-yin-hustle-fund | 40 | 37 |
| howie-diamond-pure-ventures | 18 | 6 |
| jake-chapman-marque-ventures | 16 | 2 |
| jenny-fielding-everywhere-ventures | 4 | 5 |
| jesse-middleton-flybridge | 23 | 19 |
| jillian-manus-structure-capital | 56 | 31 |
| mac-conwell | 10 | 22 |
| mark-phillips-11-tribes-ventures | 8 | 10 |
| michael-hyatt | 28 | 6 |
| mike-ma-sidecut-ventures | 11 | 3 |
| nicole-verkindt | 8 | 1 |
| paige-doherty-behind-genius-ventures | 9 | 12 |
| phil-nadel-forefront-venture-partners | 60 | 20 |
| rohit-gupta-future-communities-capital | 8 | 4 |
| sarah-downey-accomplice | 9 | 8 |
| sheel-mohnot-better-tomorrow-ventures | 17 | 2 |

### Exclusion Reasons

- `decision_not_verified`: 85
- `decision_evidence_not_preserved`: 80
- `missing_or_non_pitch_review`: 77
- `missing_in_room_decision_evidence`: 9
- `invalid_review_boundaries`: 6
- `decision_turn_removed`: 4
- `decision_not_in_room`: 3
- `decision_quote_removed`: 1

## What Was Removed

- Promotional intros and cross-episode montages, including the known Shimmur, Sudden Coffee, and Skylights exposures.
- Host/narrator turns and unattributed continuations which the original parser could merge into a VC turn.
- Post-pitch interviews, later updates, and source descriptions that can summarize outcomes.
- Explicit episode/show commentary within participant speech, including the CurieDx reference to the earlier Vital Audio investment.

Decision quotes are checked against the retained source, and evidence turn numbers are remapped. In-room decisions remain; later diligence outcomes are not substituted. Exact source and record hashes bind the installed data. Local semantic indexes were rebuilt from the retained dialogue.

This is conservative structural filtering with explicit commentary detection, not a proof against every unnamed paraphrase in live conversation. Excluded records require review before reinstatement. Missing continuation text and incomplete speaker metadata can reduce historical coverage.

## Audit and Recovery

- `outputs/pitchshow-history-clean/preparation-report.json`: coverage.
- `outputs/pitchshow-history-clean/<vc>/cleaning/*.json`: original/clean hashes, turn mappings, removals.
- `outputs/pitchshow-history-clean/<vc>/cleaning-exclusions.json`: excluded episodes and reasons.
- `outputs/pitchshow-history-clean/raw-originals/<vc>/`: original downloaded snapshots.
- `outputs/pitchshow-history-clean/install-backup-*/`: previous corpora/indexes and rollback manifest.
- `outputs/pitchshow-history-clean/original-package-manifests/`: previous installed package manifests.
- `outputs/pitchshow-history-clean/verification-report.json`: installation and preserved-run checks.

The offline preparation entry point is `uv run python scripts/prepare_pitchshow_history.py --output-root <new-directory>`. It expects the original full source corpus and matching review sources. It refuses to overwrite an existing staging directory. Build the staged local semantic indexes before `--install`; installation requires them and takes the batch execution lock. Refresh installed episode manifests after publication. The scripts used for this publication are preserved in `outputs/pitchshow-history-tools/`.

Scope is the 19 `configs/investors/<vc>/canonical.toml` configurations used by the current batch. Legacy experiment configurations and aliases (such as `phil-nadel`) remain historical snapshots and are not migrated by this publication. Use the canonical configurations for the cleaned corpora.

Raw evaluation transcripts, original A/B inputs, and saved assessment results are preserved. No assessment calls or paid reruns were made. New batch executions receive new identities because corpus/index/code hashes changed; existing run packages remain historical snapshots.

## Separate Wiki Issues

Validation: 187 tests passed across the cleaner, publisher, precedent builder/retrieval, and input firewall. Twelve installed source checks confirmed removal of the known montage excerpts. A regression test covers a held-out view with no remaining historical records, which must remain empty rather than falling back to excluded history.

See the [36-case leakage audit](results/LEAKAGE-AUDIT-36.md). Nicole/ShearShare has confirmed decision disclosure under an episode title and an unnamed-company summary. Paige/Lotus and Rohit/Mappa have suspicious target-inclusive aggregate statistics. The wiki filter also fails to propagate all `full:` source identities into derived claims. These issues are **not fixed by historical dialogue cleanup**. Existing accuracy results remain contaminated and must not be called leakage-clean.

## Review-Format Repair (19 October 2026)

The first publication wrongly excluded 85 verified records because an older review format stores its independent verdict in `verification.decisions[].pitch_decision_supported`, rather than `annotations[].pitch_evidence_verified`. The adapter now checks both formats, matches VC and exact evidence, and never overrides an explicit negative verification.

The repaired publication has **495 retained / 180 excluded** records across 19 canonical investors; Beck has 10 records instead of 1. Evidence matching, source hashes, participant filtering and commentary removal still apply. Reports, per-VC In/Out coverage, backups and indexes are under `outputs/pitchshow-history-repaired/`. Earlier reports above describe the first publication and remain for provenance. Neither completed experiment is rewritten.
