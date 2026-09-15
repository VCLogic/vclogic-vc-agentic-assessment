# Ollama Capability Report

## Outcome

The framework and local retrieval stack are operational, but this host's current
Ollama/T4 inference runtime is not fast enough to complete a full Elizabeth Yin
episode evaluation reliably. This is a runtime capability result, not a failure
of the LangGraph workflow: the same workflow completes end to end with its
deterministic provider, and its OpenAI provider is ready for later use.

The local configuration is retained as a reproducible smoke test. It should not
be used to assess VC-clone quality until the Ollama runtime throughput is fixed.

## Tested input

- Investor: Elizabeth Yin / Hustle Fund
- Episode: `135-thoras-ai-the-twin-effect`
- Audited pitch-only input: 15,250 characters
- Wiki index: 29 heading-bounded, hash-bound sections
- No actual label, complete episode transcript, or later-episode outcome was
  present in the inference package.

## Hardware and runtime observations

- Ollama reported each tested model as `100% GPU` on the T4.
- `nvidia-smi` could not initialize NVML because the installed driver and NVML
  library versions do not match.
- The Ollama client and server also reported different versions during testing.
- These environment inconsistencies are plausible contributors to the unusually
  low generation throughput; they should be repaired before judging Ollama model
  capability.

## Measured results

### Retrieval

- `qwen3-embedding:0.6b`: about 44 seconds for a representative 2,325-character
  section.
- `nomic-embed-text`: about 26 seconds for the same cold diagnostic request.
- Complete Nomic hybrid index: 29 sections in 1 minute 41.5 seconds. The index is
  saved and reused across episode runs.

### Minimal structured-generation benchmarks

| Model and mode | Input/output tokens | Wall time |
|---|---:|---:|
| `qwen3.5:4b-q4_K_M`, JSON Schema, thinking off | 25 / 22 | 48.73 s |
| `qwen3:4b`, JSON Schema, thinking off | 23 / 58 | 93.86 s |
| `qwen3:0.6b`, JSON Schema, thinking off | 29 / 73 | 88.77 s |
| `qwen3:0.6b`, JSON mode, thinking off | 30 / 19 | 22.69 s |

Even the sub-billion model generated at roughly one output token per second.
JSON mode was faster than schema-constrained decoding, so the smoke-test config
uses JSON mode and relies on the graph's strict Pydantic validation and retries.

### Full-pitch attempts

- `qwen3.5:9b` fit entirely on the GPU with a 32K context, but did not return the
  first structured response within the 15-minute client window.
- Disabling hidden thinking and bounding generation did not make the 9B call
  operational on this runtime.
- `qwen3:0.6b` in JSON mode, with the exact required schema included in the
  prompt, still returned no first-phase response after 17 minutes 46 seconds;
  the diagnostic run was stopped.

No investment prediction is reported from these attempts because no complete,
validated Phase 1 artifact existed. Reporting a partial or inferred decision
would violate the framework's evaluation contract.

## Recommended next action

1. Repair the NVIDIA/NVML and Ollama client-server version mismatches, then rerun
   the supplied smoke-test config.
2. Benchmark a trivial JSON response and require materially better throughput
   before attempting a full pitch.
3. For the research experiment now, use
   `configs/openai-elizabeth-thoras.toml`; the API key remains only in `.env` and
   is never serialized into artifacts.

The local model remains useful for infrastructure checks once runtime health is
restored. A stronger provider is still required for meaningful investor-specific
reasoning and evaluation.
