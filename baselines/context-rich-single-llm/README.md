# Context-rich single-LLM baseline

This baseline tests the supervisor's central counterfactual: what happens if one ChatGPT-compatible model receives the whole investor memory, full rationale taxonomy, audited pitch-only cut, and five most relevant target-excluded historical episodes in a single prompt?

It makes exactly one logical structured-generation request per episode. It does not run separate investigation and synthesis phases, invoke tools, repair invalid model output, or retry an episode. The OpenRouter adapter may retry transient HTTP failures up to twice; `physical_attempts` records that transport detail.

The target's actual decision is used only after inference for ordering and scoring. Each run stores the exact prompt, source hashes, selected precedents, raw provider response, validation findings, parsed result when valid, usage, cost, and summary under the configured output root.

Commands:

```bash
uv run python baselines/context-rich-single-llm/run.py preflight --pilot
uv run python baselines/context-rich-single-llm/run.py pilot
uv run python baselines/context-rich-single-llm/run.py episode --episode 20-harper-wilde
uv run python baselines/context-rich-single-llm/run.py batch
```

The full batch must be explicitly launched; the initial pilot is one observed In and one observed Out in the same deterministic round-robin order used by the agentic experiments.
