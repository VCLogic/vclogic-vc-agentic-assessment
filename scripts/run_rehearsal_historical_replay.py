#!/usr/bin/env python3
"""Run leakage-safe material-partial historical founder rehearsals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import Any
from uuid import uuid4

from dotenv import load_dotenv
from langgraph.checkpoint.sqlite import SqliteSaver

from vc_clone_graph.historical_replay_runner import load_historical_case
from vc_clone_graph.rehearsal_artifacts import RehearsalArtifactStore
from vc_clone_graph.rehearsal_canary import score_rationales
from vc_clone_graph.rehearsal_config import load_rehearsal_config
from vc_clone_graph.rehearsal_graph import RehearsalWorkflow
from vc_clone_graph.rehearsal_replay import (
    compatibility_usage,
    rank_historical_candidates,
    rank_panel_founder_statements,
    select_compatible_answer,
)
from vc_clone_graph.rehearsal_runtime import (
    build_grounded_phase2_synthesizer,
    resolve_canonical_pitch,
    resolve_investor,
)


def _add_usage(target: dict[str, int | float], source: dict[str, int | float]) -> None:
    for key, value in source.items():
        target[key] = target.get(key, 0) + value


def _retry_once(workflow: RehearsalWorkflow, session_id: str, result: dict[str, Any]):
    return workflow.retry(session_id) if result["status"] == "resumable_error" else result


def run_episode(
    *,
    root: Path,
    config_path: Path,
    output_root: Path,
    vc_slug: str,
    vc_name: str,
    episode_slug: str,
    max_questions: int,
    answer_source: str = "strict",
    preflight_only: bool = False,
) -> dict[str, Any]:
    config = load_rehearsal_config(config_path)
    case = load_historical_case(
        input_root=root / config.rehearsal.input_root,
        evaluation_root=root / "evaluation",
        vc_slug=vc_slug,
        vc_name=vc_name,
        episode_slug=episode_slug,
    )
    runtime = resolve_investor(
        config,
        vc_slug,
        target_company_aliases=case.target_company_aliases,
        excluded_episode_slug=episode_slug,
    )
    if runtime.precedents is not None and any(
        row.episode_slug == episode_slug for row in runtime.precedents.list_episodes()
    ):
        raise ValueError("target episode remains accessible in precedent retrieval")
    pitch_text = case.canary.pitch_text
    if config.classification.mode == "v41_grounded":
        if runtime.grounded_baseline is None:
            raise ValueError("historical grounded replay lacks a canonical baseline")
        pitch_text = resolve_canonical_pitch(
            runtime.grounded_baseline,
            vc_slug=vc_slug,
            workspace=root,
        )
    if preflight_only:
        baseline = runtime.grounded_baseline
        return {
            "schema_version": "historical-rehearsal-preflight-v1",
            "episode_slug": episode_slug,
            "status": "valid",
            "provider_calls": 0,
            "target_episode_excluded": True,
            "canonical_contract_version": (
                baseline.contract_version if baseline is not None else None
            ),
            "phase1_sha256": (
                baseline.investigation_sha256 if baseline is not None else None
            ),
            "phase2_sha256": baseline.decision_sha256 if baseline is not None else None,
            "canonical_pitch_bytes": len(pitch_text.encode("utf-8")),
        }

    if answer_source not in {"strict", "panel"}:
        raise ValueError("answer_source must be strict or panel")
    session_id = f"{answer_source}-replay-{episode_slug[:55]}-{uuid4().hex[:8]}"
    store = RehearsalArtifactStore.create(
        output_root, vc_slug, session_id, pitch_text
    )
    store.write_accepted(
        "session-config.json",
        {
            "evaluation_mode": "material_partial_historical_replay",
            "answer_source": answer_source,
            "episode_slug": episode_slug,
            "provider_model": config.provider.model,
            "embedding_model": config.embedding.model,
            "max_questions": max_questions,
            "candidate_semantic_threshold": 0.45,
            "max_judged_candidates_per_turn": 2,
            "target_episode_excluded": True,
            "target_company_aliases_redacted": list(case.target_company_aliases),
            "classification_mode": config.classification.mode,
            "canonical_baseline": (
                {
                    "contract_version": runtime.grounded_baseline.contract_version,
                    "phase1_sha256": runtime.grounded_baseline.investigation_sha256,
                    "phase2_sha256": runtime.grounded_baseline.decision_sha256,
                    "target_episode_excluded": True,
                }
                if runtime.grounded_baseline is not None
                else None
            ),
        },
    )
    store.write_accepted(
        "historical-canary.json", case.canary.model_dump(mode="json")
    )
    judge_usage: dict[str, int | float] = {
        "input_tokens": 0,
        "cached_input_tokens": 0,
        "output_tokens": 0,
        "cost_usd": 0.0,
        "elapsed_seconds": 0.0,
        "call_count": 0,
    }
    used_indices: set[int] = set()
    replay_turns: list[dict[str, Any]] = []
    checkpoint = output_root / f"{session_id}.sqlite"
    started = perf_counter()
    with SqliteSaver.from_conn_string(str(checkpoint)) as saver:
        phase2_synthesizer = (
            build_grounded_phase2_synthesizer(
                runtime=runtime,
                config=config,
                store=store,
                pitch=pitch_text,
            )
            if config.classification.mode == "v41_grounded"
            else None
        )
        workflow = RehearsalWorkflow(
            provider=runtime.provider,
            checkpointer=saver,
            store=store,
            investor_name=runtime.investor.display_name,
            taxonomy=runtime.taxonomy,
            retriever=runtime.retriever,
            max_questions=max_questions,
            max_output_tokens=config.rehearsal.max_output_tokens,
            reasoning_effort=config.rehearsal.reasoning_effort,
            repair_attempts=config.rehearsal.repair_attempts,
            classification_mode=config.classification.mode,
            classifier_artifact=runtime.classifier_resolution.artifact,
            classifier_fallback_reason=runtime.classifier_resolution.reason,
            minimum_questions=min(
                config.classification.minimum_questions, max_questions
            ),
            minimum_probability_impact=(
                config.classification.minimum_probability_impact
            ),
            classification_maximum_questions=min(
                config.classification.maximum_questions, max_questions
            ),
            grounded_baseline=runtime.grounded_baseline,
            phase2_synthesizer=phase2_synthesizer,
            classifier_tiebreaker=config.classification.classifier_tiebreaker,
        )
        result = _retry_once(
            workflow,
            session_id,
            workflow.start(session_id, pitch=pitch_text, vc_slug=vc_slug),
        )
        while result["status"] == "awaiting_answer":
            number = len(result.get("answers", [])) + 1
            question = store.read_json(f"turns/turn-{number:02d}/question.json")
            candidates = (
                rank_historical_candidates(
                    generated_question=question["text"],
                    generated_labels=set(question["rationale_labels"]),
                    observed_questions=case.canary.observed_questions,
                    founder_answers=case.canary.founder_answers,
                    used_indices=used_indices,
                    embedder=runtime.embedder,
                )
                if answer_source == "strict"
                else rank_panel_founder_statements(
                    generated_question=question["text"],
                    statements=case.panel_founder_statements,
                    used_indices=used_indices,
                    embedder=runtime.embedder,
                )
            )
            selection = select_compatible_answer(
                runtime.provider,
                generated_question=question["text"],
                generated_labels=question["rationale_labels"],
                candidates=candidates,
                max_candidates=2,
                max_output_tokens=2048,
                reasoning_effort=config.rehearsal.reasoning_effort,
                repair_attempts=1,
            ) if candidates else None
            turn_record: dict[str, Any] = {
                "turn": number,
                "generated_question": question["text"],
                "generated_labels": question["rationale_labels"],
                "candidate_count": len(candidates),
                "accepted": bool(selection and selection.accepted),
            }
            if selection is not None:
                turn_record["candidate_audit"] = list(selection.candidate_audit)
                turn_record["judge_attempts"] = [
                    {
                        "parsed": attempt.parsed,
                        "content": attempt.content,
                        "usage": attempt.usage.model_dump(mode="json"),
                        "elapsed_seconds": attempt.elapsed_seconds,
                        "raw_metadata": attempt.raw_metadata,
                    }
                    for attempt in selection.judge_attempts
                ]
                _add_usage(judge_usage, compatibility_usage(selection.judge_attempts))
            if selection is not None and selection.accepted:
                assert selection.observed_index is not None
                assert selection.founder_answer is not None
                used_indices.add(selection.observed_index)
                turn_record.update(
                    {
                        "observed_index": selection.observed_index,
                        "founder_answer_supplied_verbatim": selection.founder_answer,
                        "coverage": selection.compatibility.coverage,
                        "answered_clauses": list(
                            selection.compatibility.answered_clauses
                        ),
                        "unanswered_clauses": list(selection.unanswered_clauses),
                    }
                )
                result = _retry_once(
                    workflow,
                    session_id,
                    workflow.answer(session_id, selection.founder_answer),
                )
            else:
                turn_record["stopping_reason"] = "no materially compatible historical answer"
                result = _retry_once(
                    workflow, session_id, workflow.finish(session_id)
                )
            replay_turns.append(turn_record)
            store.write_accepted(
                f"turns/turn-{number:02d}/compatibility.json", turn_record
            )
            if result["status"] == "resumable_error":
                break

    evaluation: dict[str, Any] = {
        "schema_version": "historical-rehearsal-replay-v2",
        "answer_source": answer_source,
        "episode_slug": episode_slug,
        "actual_decision": case.episode.decision.status,
        "status": result["status"],
        "replay_turns": replay_turns,
        "accepted_answer_count": len(used_indices),
        "judge_usage": judge_usage,
        "elapsed_seconds": perf_counter() - started,
        "artifact_root": str(store.session_root),
    }
    if result["status"] == "complete":
        final = result["final_assessment"]
        observed = [
            {
                "taxonomy_label": row["rationale_label"],
                "direction": row["direction"],
                "salience": row["salience"],
            }
            for row in case.observed_rationales
        ]
        rationale_metrics = score_rationales(
            predicted=final["rationale_state"], observed=observed
        )
        total_usage = dict(result["usage"])
        for key in ("input_tokens", "cached_input_tokens", "output_tokens", "cost_usd"):
            total_usage[key] = total_usage.get(key, 0) + judge_usage.get(key, 0)
        evaluation.update(
            {
                "predicted_decision": final["decision"],
                "investment_likelihood": final["investment_likelihood"],
                "decision_confidence": final["decision_confidence"],
                "decision_correct": final["decision"] == case.episode.decision.status,
                "graph_usage": result["usage"],
                "total_usage": total_usage,
                "rationale_metrics": rationale_metrics.model_dump(mode="json"),
                "classification": result["founder_report"].get("classification"),
            }
        )
    else:
        evaluation["findings"] = result.get("findings", [])
    store.write_accepted("replay-evaluation.json", evaluation)
    return evaluation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--vc", required=True)
    parser.add_argument("--vc-name", required=True)
    parser.add_argument("--episode", action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-questions", type=int, default=3)
    parser.add_argument(
        "--answer-source", choices=("strict", "panel"), default="strict"
    )
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.env_file:
        load_dotenv(args.env_file.resolve(), override=False)
    summaries = []
    for episode in args.episode:
        print(f"BEGIN {episode}", flush=True)
        summary = run_episode(
            root=root,
            config_path=args.config.resolve(),
            output_root=args.output_root.resolve(),
            vc_slug=args.vc,
            vc_name=args.vc_name,
            episode_slug=episode,
            max_questions=args.max_questions,
            answer_source=args.answer_source,
            preflight_only=args.preflight_only,
        )
        summaries.append(summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "pilot-summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
