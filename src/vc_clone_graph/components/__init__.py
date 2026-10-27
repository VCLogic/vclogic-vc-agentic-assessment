"""Reusable post-assessment components.

Both components read saved Phase 1/Phase 2 artifacts only; neither imports or
changes the phase workflows. ``coverage`` asks an LLM for rationales Phase 1
did not cover; ``consolidate`` combines Phase 1, Phase 2 and coverage outputs
with deterministic rules.
"""
