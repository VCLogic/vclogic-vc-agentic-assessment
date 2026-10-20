# One In and One Out Per Remaining VC

**Audit update:** Confirmed indirect leakage was found in five runs, with three additional suspicious cases. The raw 24/36 decision score is not leakage-clean accuracy. See [the leakage audit](LEAKAGE-AUDIT-36.md). Existing outputs are preserved and tagged in the progress ledger.

This experiment selects 36 cases across the 18 installed VCs other than Jenny. It uses founders-only inputs, configured OpenRouter generation, and all three default-enabled policy safeguards.

- Frozen selection and job identities: `remaining-vcs-one-in-one-out-policy-v1.json`.
- Live progress, decisions, statuses and completed-run costs: `remaining-vcs-one-in-one-out-policy-v1-progress.json`.
- Pre-run held-out quote screen: `remaining-vcs-one-in-one-out-policy-v1-final-screen.json`.
- Automatically refreshed rationale and decision metrics: report referenced by `automated-evaluation/latest.json`; match its rows to the frozen manifest's job IDs.
- Local execution log: `outputs/pitchshow-evaluation/experiments/remaining-vcs-one-in-one-out-policy-v1.log`.

Selection prefers original numbered episodes without prior VC/pitch results, then lowest episode number. Seven selected cases have legacy results with different input/settings contracts; those remain separate evaluations. Nicole's episode 37 was excluded after finding a target quote in her wiki without the company name, and episode 38 was selected instead. Predictions did not influence selection.

The batch preserves completed results and does not automatically retry failed jobs. Resume the same frozen experiment with:

```bash
uv run python outputs/pitchshow-evaluation/experiments/run_remaining_vcs.py
```

The runner uses the shared batch lock and refuses a changed selection. Source/configuration changes invalidate a pending job rather than silently running a different experiment. Local outputs contain the runner and logs; they are not versioned package entry points.

## Held-Out Evidence

Later references are allowed. Target-pitch decision information is not. The wiki view now excludes whole target-containing sections and sections connected through their evidence citations; canonical wiki files remain unchanged. This conservatively removes some unrelated context as well. The new exclusion has regression coverage for investment amounts and derived claims. It is stricter than the view used for the earlier Jenny reruns.

All 36 selected cases passed an exact ten-word target-rationale quote overlap check against the filtered wiki. This does not prove absence of paraphrased contamination. References remain automated candidates, and these selected cases are a diagnostic sample, not an overall accuracy estimate. Reported completed-run costs may omit charges incurred by failed attempts.
