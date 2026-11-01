# Proposal evaluation

Reproduces every result in the 30 October proposal (`reports/VCLogic 30 Oct NN ee.docx`): Tables 1, 2 and A4–A11, Figures A2–A4, and the numbers quoted in the text. It makes no model calls.

## Command

From the repository root:

```bash
uv run --with matplotlib python proposal-evaluation/evaluate.py
```

The full run takes a few minutes, mostly for the 2,000 bootstrap resamples. Add `--quick` to use 200 resamples; the point estimates stay the same and the intervals are approximate.

## Output (`results/`)

| File | Content |
|---|---|
| `results.md` | Every table, in the order of the proposal. It ends with **Check against the draft**: each number in the draft next to the value computed here, marked `ok` or `differs`. |
| `tables/*.csv` | One CSV per table and per figure. |
| `figures/*.png` | Figures A2 (recall by dimension), A3 (share called In per investor) and A4 (backed pitches among the top k). |

## Data (`data/`)

Coded records only: rationale labels, directions, salience, confidence, In/Out calls and likelihoods. No transcripts, pitches or quotes are included.

| File | Content |
|---|---|
| `decisions.jsonl` | The 400 investor–pitch decisions. Each has the actual decision on air; the investigator agent's rationales, including those added by the taxonomy review (these give the weighted rationale score); the decision agent's final rationales, call and likelihood; and the investor's reference rationales (317 decisions). |
| `taxonomy.json` | Rationale label → dimension. |
| `reference_second_coding.jsonl` | The second, independent coding of the same on-air statements, for the agreement figures. |
| `funnel.jsonl`, `funnel_counts.json` | Every episode record per investor, from 750 records to 400 assessed decisions (Table A4). |
| `no_decision_responses.csv` | Category of the 73 on-panel responses without an on-air In or Out. |
| `single_call.jsonl`, `single_call_run.json` | The single-call baseline on the 200 decisions it completed (Table A10). |
| `sol_pilots.json` | The two GPT-6 Sol pilots of 16 decisions (Table A11). |
| `memory_sizes.json` | Evidence excerpts and context entries per Investment Memory. |

`build_data.py` created these files from the local evaluation records (`evaluation/`, `outputs/`, `rationale_annotation/`, `inputs/wiki/`). It does not need to be run again; it is kept to show where each number comes from:

```bash
uv run python proposal-evaluation/build_data.py
```

## Privacy

`data/` and `results/` hold investor-specific records, so they are listed in `.gitignore` and stay on this machine, like the rest of the evaluation data (see the README's "Investor data and privacy" section). Only this README and the two scripts can be committed.

## Methods in brief

1. **Weighted rationale score.** The sum, over the investigator's rationales, of direction (+1 / −1 / 0) × salience (1 primary, 0.5 secondary) × confidence. A pitch is In when its score is in the investor's top 30% of their other pitches; investors with fewer than eight other pitches use all pitches.
2. **Rationales** compare the decision agent's final list with the investor's reference rationales, at the level of dimension, rationale, and rationale with direction.
3. **Intervals.** Whole episodes are resampled with replacement (seed 0). Tables 2, A5, A6, A9 and A10 use 2,000 resamples (`numpy.random.default_rng(0)`). Table A7 repeats the pre-specified tests, which used 1,000 resamples (`random.seed(0)`). Every model in Table A7 is trained leaving out one episode at a time.
4. **Not covered.** Table A3 (sources collected per investor) comes from the VC Trace Collector's records, which are not in this repository.
