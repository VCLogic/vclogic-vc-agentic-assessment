# Memory-swap pilot: results

**Status: PILOT, exploratory.** Sample, conditions and criteria were pre-registered in `criteria.json` (commit f4eeff6) before any model call.

## Summary

1. Giving the twin some investor evidence clearly changes what it says. Without evidence it misses about one in six dimensions the investor raised, almost never raises investor-fit reasons, and sounds more like the other twins.
2. In this pilot, the twin was not detectably closer to its own investor with its own evidence than with another panelist's evidence, so the primary criteria were not met.

## Sample and runs

1. 20 split panels, 66 decisions (18 investors), drawn as pre-registered. Each decision was run in four conditions: OWN, SWAP, NAME_ONLY and GENERIC. OWN_ANON was not run.
2. 251 of 264 runs completed. The 13 missing runs are ones the pipeline gave up on after its own retries; they were not retried at extra cost. OWN 0 missing, SWAP 1, NAME_ONLY 9, GENERIC 3. Eleven of the 13 are in the two no-evidence conditions: the investigator must cite investor evidence for each reason, and with only an empty placeholder it sometimes fails to do so.
3. The table uses the 53 decisions (19 panels) completed in every condition. A check on all 65 decisions completed in both OWN and SWAP is given below.
4. Inference: paired differences against OWN, 2,000 bootstrap samples over panels, seed 0, 95% percentile intervals. Balanced accuracy uses each investor's 70th-percentile cutoff from the main evaluation, the same in every condition.

## Results

Each cell gives the value, then the difference from OWN in points with its 95% interval. The MAIN column shows the main evaluation's outputs on the same 53 decisions.

| Measure | OWN | SWAP | NAME_ONLY | GENERIC | MAIN (same decisions) |
|---|---|---|---|---|---|
| dimension recall | 96.0% | 96.9% (+0.9; -1.3 to +3.0) | 79.4% (-16.6; -21.1 to -12.8) | 82.1% (-13.9; -16.8 to -10.7) | 95.1% (OWN−MAIN +0.9; -0.9 to +3.4) |
| dimension precision | 58.6% | 59.2% (+0.5; -0.7 to +2.0) | 55.5% (-3.1; -5.1 to -1.3) | 55.3% (-3.3; -4.9 to -1.7) | 57.6% (OWN−MAIN +1.0; -0.5 to +2.3) |
| dimension F1 | 72.8% | 73.5% (+0.7; -0.5 to +2.2) | 65.3% (-7.5; -9.9 to -5.1) | 66.1% (-6.7; -8.7 to -4.7) | 71.7% (OWN−MAIN +1.0; -0.2 to +2.3) |
| rationale recall | 59.5% | 58.0% (-1.5; -6.4 to +3.8) | 52.5% (-7.0; -12.5 to -2.3) | 54.2% (-5.2; -10.1 to +0.0) | 63.0% (OWN−MAIN -3.5; -6.6 to -0.8) |
| rationale precision | 24.2% | 22.7% (-1.5; -3.2 to +0.5) | 25.6% (+1.5; -1.1 to +3.7) | 26.1% (+1.9; -0.5 to +4.4) | 25.0% (OWN−MAIN -0.8; -2.0 to +0.2) |
| rationale F1 | 34.4% | 32.6% (-1.8; -4.2 to +0.9) | 34.4% (+0.1; -3.3 to +2.8) | 35.2% (+0.9; -2.1 to +4.0) | 35.8% (OWN−MAIN -1.4; -2.8 to -0.0) |
| rationale + direction recall | 30.4% | 28.5% (-1.9; -5.0 to +2.2) | 26.5% (-3.9; -8.0 to -0.4) | 27.6% (-2.8; -6.5 to +1.0) | 31.1% (OWN−MAIN -0.7; -4.2 to +2.0) |
| rationale + direction precision | 15.5% | 14.0% (-1.5; -3.3 to +0.3) | 16.2% (+0.7; -1.6 to +2.7) | 16.7% (+1.2; -1.4 to +3.5) | 15.5% (OWN−MAIN +0.0; -1.7 to +1.5) |
| rationale + direction F1 | 20.5% | 18.8% (-1.8; -3.9 to +0.6) | 20.1% (-0.4; -3.2 to +2.0) | 20.8% (+0.3; -2.8 to +3.0) | 20.7% (OWN−MAIN -0.1; -2.3 to +1.7) |
| decision-linked recall | 62.9% | 58.3% (-4.5; -14.4 to +4.8) | 44.7% (-18.2; -27.1 to -9.3) | 51.5% (-11.4; -21.1 to -2.0) | 63.6% (OWN−MAIN -0.8; -7.8 to +4.4) |
| investor-fit recall | 31.5% | 22.2% (-9.3; -26.4 to +10.0) | 1.9% (-29.6; -40.4 to -20.4) | 3.7% (-27.8; -40.0 to -16.7) | 35.2% (OWN−MAIN -3.7; -11.1 to +5.0) |
| target-investor identification | 46.6% | 45.5% (-1.1; -14.5 to +13.5) | 47.7% (+1.1; -10.3 to +12.5) | 56.8% (+10.2; -4.3 to +25.5) | 55.7% (OWN−MAIN -9.1; -18.7 to +0.0) |
| overlap between twins | 67.7% | 71.7% (+4.0; -0.4 to +9.1) | 77.8% (+10.1; +4.7 to +16.5) | 78.6% (+10.9; +6.5 to +16.0) | 70.0% (OWN−MAIN -2.3; -6.3 to +1.4) |
| within-panel AUC (weighted score) | 0.46 | 0.64 (+0.18; -0.12 to +0.46) | 0.64 (+0.18; -0.04 to +0.44) | 0.43 (-0.04; -0.39 to +0.32) | 0.57 (OWN−MAIN -0.11; -0.39 to +0.16) |
| balanced accuracy (weighted score, main cutoffs) | 60.1% | 56.6% (-3.5; -15.4 to +6.8) | 67.3% (+7.2; -1.5 to +16.4) | 56.6% (-3.5; -16.0 to +7.1) | 56.6% (OWN−MAIN +3.5; -9.4 to +16.7) |
| rationales per pitch | 18.6 | 18.8 (+0.2; -1.3 to +1.5) | 15.1 (-3.5; -4.3 to -2.5) | 15.4 (-3.2; -4.0 to -2.4) | 19.0 (OWN−MAIN -0.4; -1.1 to +0.4) |
| share in favour | 50.9% | 47.4% (-3.4; -7.4 to +0.4) | 58.8% (+7.9; +4.4 to +11.6) | 57.9% (+7.0; +3.1 to +11.7) | 51.4% (OWN−MAIN -0.6; -3.7 to +2.4) |

Overlap between the investors themselves on the same pitches: 39.8%.

## Criteria

1. **Primary (OWN − SWAP > 0, interval excludes 0): not met.**
   1. Target-investor identification: OWN 46.6% vs SWAP 45.5%, difference +1.1 points (−13.5 to +14.5).
   2. Decision-linked recall: OWN 62.9% vs SWAP 58.3%, difference +4.5 points (−4.8 to +14.4).
   3. On all 65 decisions completed in both OWN and SWAP, the differences are +0.9 points (−10.0 to +11.2) and +3.7 points (−5.1 to +12.0).
2. **Secondary.**
   1. OWN > NAME_ONLY on decision-linked recall: met (+18.2 points, 9.3 to 27.1).
   2. OWN > GENERIC on decision-linked recall: met (+11.4 points, 2.0 to 21.1).
   3. OWN > NAME_ONLY or GENERIC on target-investor identification: not met. GENERIC is 10.2 points higher than OWN, with an interval that includes 0.
   4. Rationale F1 shows no clear difference between conditions. Dimension F1 is lower without evidence (−7.5 and −6.7 points).
   5. Within-panel AUC: no difference is clear (intervals span about ±0.4).
   6. Overlap between twins on the same pitch is higher without evidence (+10.1 and +10.9 points); SWAP is +4.0 (−0.4 to +9.1).

## What the evidence changes

1. Investor-fit reasons nearly vanish without evidence: 31.5% recall with OWN, 1.9% with NAME_ONLY and 3.7% with GENERIC. SWAP gives 22.2%, but the interval is wide (−26.4 to +10.0 points).
2. Without evidence the twin gives fewer reasons (about 15 instead of 19 per pitch), and more of them are in favour (58–59% instead of 51%).
3. SWAP is close to OWN on every measure. Another investor's evidence works about as well as the investor's own.

## OWN vs MAIN check

OWN reruns the main pipeline with the same inputs, so differences from MAIN show run-to-run variation.

1. Most measures agree within about 1–2 points.
2. Rationale recall is 3.5 points lower in OWN (−6.6 to −0.8), and rationale F1 is 1.4 points lower (−2.8 to −0.0).
3. Target-investor identification is 9.1 points lower in OWN (−18.7 to 0.0): 46.6% vs 55.7% on these decisions. With about 50 decisions, this measure moves by several points between runs of the same pipeline, so the pilot cannot resolve differences of that size.
4. Within-panel AUC (0.46 vs 0.57) and balanced accuracy (60.1% vs 56.6%) also vary widely between runs.

## Notes

1. **Donor names in SWAP.** The donor's evidence names the donor investor, while the twin is told it is the target. The twin therefore sees a mismatch between its name and its evidence.
2. **Leakage.** In every condition the current episode was excluded from prior cases. Investment Memory sections naming the pitched company were redacted at run time. This was checked in all 16 runs whose supplied memory named the company: between 1 and 132 sections were redacted in each.
3. **Placeholders.** NAME_ONLY and GENERIC used an explicit empty Investment Memory, with prior cases switched off and no profiles in the lens. The empty inputs are listed per run in `results.json`.
4. **Cost.** USD 27.49 in total, including runs that failed, with no runs skipped for budget. Completed runs cost USD 0.13 (OWN), 0.14 (SWAP), 0.08 (NAME_ONLY) and 0.07 (GENERIC) per decision.
5. **Model and settings.** `openai/gpt-5.6-luna` via OpenRouter. Each decision used the main evaluation's own run configuration and the default coverage and contrastive lens settings.
6. **Code.** `run_pilot.py` (runner), `evaluate.py` (measures copied from `outputs/final-solution/report.py`), `sample.py` (pre-registered sample).
