#!/usr/bin/env python3
"""Evaluate rehearsal-question fidelity locally without generation calls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from vc_clone_graph.question_memory import QuestionMemory
from vc_clone_graph.rehearsal_canary import HistoricalRehearsalCanary, score_question
from vc_clone_graph.rehearsal_config import load_rehearsal_config
from vc_clone_graph.rehearsal_question_evaluation import (
    EpisodeQuestionEvaluation,
    aggregate_episode,
    score_generated_question,
    write_evaluation_outputs,
)
from vc_clone_graph.rehearsal_runtime import (
    _embedding_provider,
    _load_precedents,
    _taxonomy,
)


def discover_session_roots(root: Path) -> tuple[Path, ...]:
    """Find complete historical rehearsal sessions deterministically."""
    return tuple(
        sorted(
            path.parent
            for path in Path(root).rglob("session-config.json")
            if (path.parent / "historical-canary.json").is_file()
            and tuple(path.parent.glob("turns/turn-*/question.json"))
        )
    )


def _nearest_similarity(text: str, observed, embedder) -> float | None:
    values = [
        score_question(
            generated=text,
            generated_labels=set(),
            observed=row.text,
            observed_labels=set(),
            embedder=embedder,
        ).semantic_similarity
        for row in observed
    ]
    return max(values) if values else None


def evaluate(
    *,
    root: Path,
    config_path: Path,
    vc_slug: str,
    vc_name: str,
    session_root: Path,
    output_root: Path,
) -> tuple[EpisodeQuestionEvaluation, ...]:
    config = load_rehearsal_config(config_path)
    embedder = _embedding_provider(config)
    input_root = (root / config.rehearsal.input_root).resolve()
    taxonomy = _taxonomy(input_root / config.rehearsal.taxonomy_path)
    taxonomy_parents = {row["label"]: row["coarse_parent"] for row in taxonomy}
    aliases = (vc_name, vc_name.split()[0])
    memories: dict[str, QuestionMemory] = {}
    evaluations: list[EpisodeQuestionEvaluation] = []
    ablations: list[dict[str, Any]] = []
    for session in discover_session_roots(session_root):
        metadata = json.loads((session / "session-config.json").read_text())
        episode_slug = str(metadata["episode_slug"])
        canary = HistoricalRehearsalCanary.model_validate_json(
            (session / "historical-canary.json").read_text(encoding="utf-8")
        )
        if episode_slug != canary.episode_slug:
            raise ValueError(f"session/canary episode mismatch: {session}")
        if episode_slug not in memories:
            precedents = _load_precedents(
                config,
                input_root,
                vc_slug,
                embedder,
                excluded_episode_slug=episode_slug,
                excluded_aliases=(),
            )
            memories[episode_slug] = QuestionMemory(
                precedents.question_archetypes(aliases) if precedents else (),
                embedder,
                taxonomy=taxonomy,
                settings=config.question_memory,
            )
        memory = memories[episode_slug]
        records = []
        prior: list[str] = []
        for question_path in sorted(session.glob("turns/turn-*/question.json")):
            question = json.loads(question_path.read_text(encoding="utf-8"))
            record = score_generated_question(
                question_id=str(question["question_id"]),
                generated_text=str(question["text"]),
                generated_labels=tuple(question.get("rationale_labels", [])),
                observed=canary.observed_questions,
                embedder=embedder,
                prior_generated=tuple(prior),
                taxonomy_parents=taxonomy_parents,
            )
            records.append(record)
            audit_path = question_path.with_name("question-archetypes.json")
            audit = (
                json.loads(audit_path.read_text(encoding="utf-8"))
                if audit_path.is_file()
                else {}
            )
            query = str(audit.get("query") or question["text"])
            requested_labels = tuple(
                audit.get("requested_rationale_labels")
                or question.get("rationale_labels", [])
            )
            for policy in ("semantic", "hybrid"):
                hits = memory.search(
                    query,
                    rationale_labels=requested_labels,
                    prior_questions=tuple(prior),
                    policy=policy,
                )
                top = hits[0] if hits else None
                ablations.append(
                    {
                        "session_id": session.name,
                        "replay_mode": metadata.get("answer_source", "unspecified"),
                        "episode_slug": episode_slug,
                        "question_id": question["question_id"],
                        "policy": policy,
                        "candidate_count": len(hits),
                        "top_archetype_id": top.archetype_id if top else None,
                        "top_archetype_episode": top.episode_slug if top else None,
                        "top_archetype_text": top.text if top else None,
                        "top_semantic_score": top.semantic_similarity if top else None,
                        "top_hybrid_score": top.hybrid_score if top else None,
                        "nearest_observed_similarity": (
                            _nearest_similarity(top.text, canary.observed_questions, embedder)
                            if top
                            else None
                        ),
                        "atomic": bool(top and not top.atomicity_findings),
                        "target_excluded": bool(
                            top is None or top.episode_slug != episode_slug
                        ),
                    }
                )
            prior.append(str(question["text"]))
        evaluations.append(
            aggregate_episode(
                episode_slug=episode_slug,
                session_id=session.name,
                replay_mode=str(metadata.get("answer_source", "unspecified")),
                records=tuple(records),
                observed_questions=canary.observed_questions,
            )
        )
    if not evaluations:
        raise ValueError(f"no complete historical rehearsal sessions found: {session_root}")
    write_evaluation_outputs(
        evaluations=tuple(evaluations),
        ablations=tuple(ablations),
        output_root=output_root,
    )
    return tuple(evaluations)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--vc", required=True)
    parser.add_argument("--vc-name", required=True)
    parser.add_argument("--session-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    rows = evaluate(
        root=root,
        config_path=args.config.resolve(),
        vc_slug=args.vc,
        vc_name=args.vc_name,
        session_root=args.session_root.resolve(),
        output_root=args.output_root.resolve(),
    )
    print(
        json.dumps(
            [row.model_dump(mode="json") for row in rows],
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
