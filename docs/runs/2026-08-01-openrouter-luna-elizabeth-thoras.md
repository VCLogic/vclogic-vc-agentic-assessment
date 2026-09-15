# OpenRouter Luna example: Elizabeth Yin / Thoras AI

## Configuration

- Investor memory: Elizabeth Yin / Hustle Fund
- Episode: `135-thoras-ai-the-twin-effect`
- Generation model: `openai/gpt-5.6-luna` through OpenRouter
- Retrieval embeddings: local Ollama `nomic-embed-text`
- Bounds: at most two Phase 1 passes and two Phase 2 passes
- Input: audited pitch-only transcript and the investor wiki; actual outcome excluded

## Accepted result

Attempt 2 completed both phases in 42 seconds. Phase 1 identified 14 rationales and
was accepted as saturated. Phase 2 returned:

- recommendation: `Out`
- investment likelihood: `0.42`
- decision confidence: `0.75`
- human-review ranking score: `0.74`
- check tier: `no_check_tier`

The observed on-panel decision was `In`: Elizabeth offered $50,000. The run therefore
missed the binary decision while still ranking the pitch as a relatively high-priority
candidate for human review. The synthesis gave substantial weight to the absence of
live customer-environment validation, repeatable pilot conversion, a narrow initial
segment, demonstrated differentiation, and established enterprise economics. It did
recognize the founders' domain experience, concrete problem, paid pilots, customer
orientation, and investor demand.

Accepted-attempt usage:

- input tokens: 19,063
- output tokens: 4,203
- cached input tokens: 0
- OpenRouter-reported cost: $0.00490445

The accepted frozen artifacts are under
`outputs/openrouter-luna-elizabeth-thoras-attempt-2/135-thoras-ai-the-twin-effect/`.

## Development attempts

Attempt 1 failed strict mechanical validation after two passes because Luna emitted a
mistyped wiki chunk identifier and an inexact episode slug. Run-specific JSON Schema
constraints now prevent those identifier errors.

Attempt 3 retained the complete planner and investigator trace but stopped after Phase 1
because the original validator required literal pitch substrings. Inspection showed that
the rejected evidence consisted of close, faithful paraphrases. The validator now accepts
conservative near-matches while continuing to reject unrelated claims. Attempt 3's second
Phase 1 candidate passes the corrected validator; it was not replayed through Phase 2, so
no additional paid call was made.

Total OpenRouter-reported development cost across all three attempts was $0.02552532.

## Interpretation

This is a pipeline smoke test, not a performance estimate. It demonstrates that OpenRouter
Luna can perform both phases, use local wiki retrieval iteratively, return schema-constrained
artifacts, and report usage and cost. It also reproduces the substantive challenge seen in
earlier experiments: the system can identify an eventual investment as worth reviewing
while applying a stricter binary investment threshold than the observed investor.
