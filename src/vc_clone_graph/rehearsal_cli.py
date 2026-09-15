"""Command-line interface for interactive founder rehearsal sessions."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
from typing import Any, Callable, Sequence
from uuid import uuid4

from langgraph.checkpoint.sqlite import SqliteSaver

from .rehearsal_artifacts import RehearsalArtifactStore
from .rehearsal_bootstrap import (
    build_or_load_canonical_baseline,
    import_canonical_baseline,
)
from .rehearsal_config import RehearsalConfig, load_rehearsal_config
from .rehearsal_depth import effective_question_limit
from .rehearsal_graph import RehearsalWorkflow
from .rehearsal_grounding import load_canonical_baseline_from_run
from .rehearsal_runtime import (
    build_grounded_phase2_synthesizer,
    list_investors,
    resolve_investor,
)


def locate_session(output_root: Path, session_id: str) -> Path:
    matches = sorted(Path(output_root).resolve().glob(f"*/{session_id}"))
    matches = [path for path in matches if (path / "manifest.json").is_file()]
    if not matches:
        raise ValueError(f"unknown rehearsal session: {session_id}")
    if len(matches) > 1:
        raise ValueError(f"ambiguous rehearsal session ID: {session_id}")
    return matches[0]


def _bullets(values: Sequence[str]) -> str:
    return "\n".join(f"- {value}" for value in values) if values else "- None recorded"


def markdown_report(report: dict[str, Any]) -> str:
    initial = report["initial_assessment"]
    final = report["final_assessment"]
    classification = report.get("classification")
    classification_section = ""
    if classification and classification.get("status") == "available":
        rows = classification.get("snapshots", [])
        trajectory = " → ".join(
            f"{row['stage']} {row['probability_in']:.0%} ({row['predicted_decision']})"
            for row in rows
        )
        classification_section = f"""
## Classification-informed trajectory

- Model: `{classification.get('model_version')}`
- Trajectory: {trajectory}
- This classifier endpoint is preserved separately from the simulated investor judgment.
"""
    elif classification:
        classification_section = f"""
## Classification-informed trajectory

- Unavailable; the interview used rationale-only fallback.
- Reason: {classification.get('fallback_reason')}
"""
    grounded = report.get("grounded")
    grounded_section = ""
    if grounded:
        baseline = grounded["baseline"]
        changes = grounded.get("rationale_changes", [])
        change_lines = _bullets(
            [
                f"{row['taxonomy_label']}: {row['change']} (answers: {', '.join(row['answer_ids'])})"
                for row in changes
            ]
        )
        grounded_section = f"""
## Canonical baseline versus revised assessment

- Baseline: **{baseline['decision']}** ({baseline['investment_likelihood']:.0%}) using contract `{baseline['contract_version']}`
- After founder answers: **{grounded['final_decision']}** ({grounded['final_investment_likelihood']:.0%})
- Decision changed: **{'Yes' if grounded['decision_changed'] else 'No'}**

### Answer-linked rationale changes

{change_lines}
"""
    return f"""# Founder Rehearsal Report

> {report['disclosure']}

## Simulated investor judgment

- Initial assessment: **{initial['decision']}** ({initial['investment_likelihood']:.0%} likelihood; {initial['decision_confidence']:.0%} confidence)
- Final assessment: **{final['decision']}** ({final['investment_likelihood']:.0%} likelihood; {final['decision_confidence']:.0%} confidence)
- Justification: {final['decision_justification']}

### Unresolved uncertainties

{_bullets(final.get('unresolved_uncertainties', []))}

### Reversal conditions

{_bullets(final.get('reversal_conditions', []))}
{classification_section}
{grounded_section}

## Founder coaching

### Pitch improvements

{_bullets(report.get('pitch_improvement_suggestions', []))}

### Reflection prompts

{_bullets(report.get('founder_reflection_prompts', []))}

## Usage

```json
{json.dumps(report.get('usage', {}), indent=2, sort_keys=True)}
```
"""


def _session_metadata(store: RehearsalArtifactStore) -> dict[str, Any]:
    return store.read_json("session-config.json")


def _build_workflow(
    config: RehearsalConfig,
    store: RehearsalArtifactStore,
    saver: Any,
    runtime: Any | None = None,
    progress_callback: Callable[[str, dict[str, Any]], Any] | None = None,
) -> RehearsalWorkflow:
    manifest = store.read_json("manifest.json")
    metadata = _session_metadata(store)
    session_question_limit = max(
        1,
        min(8, int(metadata.get("max_questions", config.rehearsal.max_questions))),
    )
    aliases = tuple(str(value) for value in metadata.get("target_company_aliases", []))
    grounded_baseline = None
    canonical_run_path = metadata.get("canonical_run_path")
    if config.classification.mode == "v41_grounded" and canonical_run_path:
        grounded_baseline = load_canonical_baseline_from_run(
            workspace=config.workspace,
            run_root=Path(str(canonical_run_path)),
            canonical_vc_slug=str(manifest["vc_slug"]),
            expected_run_vc_slug=str(manifest["vc_slug"]),
            expected_episode_slug=str(metadata["episode_slug"]),
            expected_pitch_sha256=str(metadata["canonical_pitch_sha256"]),
        )
    if runtime is None:
        runtime = resolve_investor(
            config,
            str(manifest["vc_slug"]),
            target_company_aliases=aliases,
            excluded_episode_slug=metadata.get("episode_slug"),
            grounded_baseline=grounded_baseline,
        )
    phase2_synthesizer = (
        build_grounded_phase2_synthesizer(
            runtime=runtime,
            config=config,
            store=store,
            pitch=store.pitch_path.read_text(encoding="utf-8"),
        )
        if config.classification.mode == "v41_grounded"
        else None
    )
    return RehearsalWorkflow(
        provider=runtime.provider,
        checkpointer=saver,
        store=store,
        investor_name=runtime.investor.display_name,
        taxonomy=runtime.taxonomy,
        retriever=runtime.retriever,
        max_questions=session_question_limit,
        max_output_tokens=config.rehearsal.max_output_tokens,
        reasoning_effort=config.rehearsal.reasoning_effort,
        repair_attempts=config.rehearsal.repair_attempts,
        classification_mode=config.classification.mode,
        classifier_artifact=runtime.classifier_resolution.artifact,
        classifier_fallback_reason=runtime.classifier_resolution.reason,
        minimum_questions=config.classification.minimum_questions,
        minimum_probability_impact=(
            config.classification.minimum_probability_impact
        ),
        classification_maximum_questions=min(
            config.classification.maximum_questions,
            session_question_limit,
        ),
        grounded_baseline=runtime.grounded_baseline,
        phase2_synthesizer=phase2_synthesizer,
        classifier_tiebreaker=config.classification.classifier_tiebreaker,
        progress_callback=progress_callback,
    )


def _checkpoint(config: RehearsalConfig) -> Path:
    path = config.resolve_path(config.rehearsal.checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def command_investors(config: RehearsalConfig) -> None:
    rows = [
        {
            "vc_slug": row.vc_slug,
            "display_name": row.display_name,
            "firm": row.firm,
            "role": row.role,
        }
        for row in list_investors(config.resolve_path(config.rehearsal.input_root))
    ]
    print(json.dumps(rows, indent=2, sort_keys=True))


def command_start(
    config: RehearsalConfig,
    args: argparse.Namespace,
    progress_callback: Callable[[str, dict[str, Any]], Any] | None = None,
) -> None:
    pitch = (
        args.pitch.read_text(encoding="utf-8") if args.pitch is not None else args.text
    )
    if not isinstance(pitch, str) or not pitch.strip():
        raise ValueError("pitch text must not be empty")
    pitch = pitch.strip() + "\n"
    session_id = args.session or str(uuid4())
    aliases = tuple(args.company or ())
    rehearsal_depth = getattr(args, "rehearsal_depth", "standard")
    question_limit = effective_question_limit(
        rehearsal_depth, config.rehearsal.max_questions
    )
    if config.classification.mode == "v41_grounded":
        if not args.build_canonical_baseline and args.canonical_baseline is None:
            raise ValueError(
                "live v41_grounded rehearsal requires explicit authorization: "
                "use --build-canonical-baseline or --canonical-baseline PATH"
            )
        if not aliases:
            raise ValueError(
                "grounded live rehearsal requires at least one --company alias"
            )
    session_root = (
        config.resolve_path(config.rehearsal.output_root) / args.vc / session_id
    )
    if session_root.exists():
        store = RehearsalArtifactStore.open(session_root)
        verified = store.verify()
        if verified["pitch_sha256"] != sha256(pitch.encode("utf-8")).hexdigest():
            raise ValueError("existing rehearsal session contains a different pitch")
        if (session_root / "session-config.json").is_file():
            raise ValueError("rehearsal session has already started; use answer or finish")
    else:
        store = RehearsalArtifactStore.create(
            config.resolve_path(config.rehearsal.output_root), args.vc, session_id, pitch
        )
    grounded_baseline = None
    episode_slug = None
    if config.classification.mode == "v41_grounded":
        workspace = config.workspace
        if args.build_canonical_baseline:
            episode_slug = f"live-{sha256(f'{args.vc}:{session_id}'.encode()).hexdigest()[:16]}"
            grounded_baseline = build_or_load_canonical_baseline(
                workspace=workspace,
                rehearsal_config=config,
                session_root=store.session_root,
                vc_slug=args.vc,
                episode_slug=episode_slug,
                pitch=pitch,
                target_company_aliases=aliases,
            )
        else:
            pitch_digest = sha256(pitch.encode("utf-8")).hexdigest()
            grounded_baseline = import_canonical_baseline(
                workspace=workspace,
                source_run=args.canonical_baseline,
                session_root=store.session_root,
                expected_vc_slug=args.vc,
                expected_pitch_sha256=pitch_digest,
            )
            episode_slug = grounded_baseline.episode_slug
    runtime = resolve_investor(
        config,
        args.vc,
        target_company_aliases=aliases,
        excluded_episode_slug=episode_slug,
        grounded_baseline=grounded_baseline,
    )
    store.write_accepted(
        "session-config.json",
        {
            "vc_slug": args.vc,
            "investor_name": runtime.investor.display_name,
            "provider_kind": config.provider.kind,
            "provider_model": config.provider.model,
            "embedding_kind": config.embedding.kind,
            "embedding_model": config.embedding.model,
            "embedding_revision": config.embedding.revision,
            "target_company_aliases": list(aliases),
            "episode_slug": episode_slug,
            "canonical_run_path": (
                str(grounded_baseline.run_root)
                if grounded_baseline is not None
                else None
            ),
            "canonical_pitch_sha256": (
                json.loads(
                    (grounded_baseline.run_root / "input-provenance.json").read_text(
                        encoding="utf-8"
                    )
                ).get("pitch_sha256")
                if grounded_baseline is not None
                else None
            ),
            "rehearsal_depth": rehearsal_depth,
            "max_questions": question_limit,
            "classification_mode": config.classification.mode,
            "classifier_status": runtime.classifier_resolution.status,
            "classifier_artifact_id": (
                runtime.classifier_resolution.artifact.artifact_id
                if runtime.classifier_resolution.artifact is not None
                else None
            ),
        },
    )
    with SqliteSaver.from_conn_string(str(_checkpoint(config))) as saver:
        workflow = _build_workflow(
            config, store, saver, runtime, progress_callback=progress_callback
        )
        result = workflow.start(session_id, pitch=pitch, vc_slug=args.vc)
    result["disclosure"] = (
        f"Simulation of {runtime.investor.display_name}; not the real investor and "
        "not endorsed by them."
    )
    print(json.dumps(result, indent=2, sort_keys=True))


def _resume(
    config: RehearsalConfig,
    session_id: str,
    action: str,
    text: str | None = None,
    progress_callback: Callable[[str, dict[str, Any]], Any] | None = None,
) -> None:
    root = locate_session(config.resolve_path(config.rehearsal.output_root), session_id)
    store = RehearsalArtifactStore.open(root)
    store.verify()
    with SqliteSaver.from_conn_string(str(_checkpoint(config))) as saver:
        workflow = _build_workflow(
            config, store, saver, progress_callback=progress_callback
        )
        if action == "answer":
            assert text is not None
            result = workflow.answer(session_id, text)
        elif action == "finish":
            result = workflow.finish(session_id)
        elif action == "retry":
            result = workflow.retry(session_id)
        else:  # pragma: no cover - internal dispatch invariant
            raise ValueError(f"unknown resume action: {action}")
    print(json.dumps(result, indent=2, sort_keys=True))


def command_report(
    config: RehearsalConfig, session_id: str, output_format: str
) -> None:
    root = locate_session(config.resolve_path(config.rehearsal.output_root), session_id)
    store = RehearsalArtifactStore.open(root)
    store.verify()
    report = store.read_json("founder-report.json")
    print(
        markdown_report(report)
        if output_format == "markdown"
        else json.dumps(report, indent=2, sort_keys=True)
    )


def command_verify(config: RehearsalConfig, session_id: str) -> None:
    root = locate_session(config.resolve_path(config.rehearsal.output_root), session_id)
    print(
        json.dumps(
            RehearsalArtifactStore.open(root).verify(), indent=2, sort_keys=True
        )
    )


def command_compare(
    config: RehearsalConfig, sessions: Sequence[str], output_format: str
) -> None:
    from .rehearsal_compare import compare_sessions, comparison_csv

    paths = [
        locate_session(config.resolve_path(config.rehearsal.output_root), session)
        for session in sessions
    ]
    comparison = compare_sessions(paths)
    print(
        comparison_csv(comparison)
        if output_format == "csv"
        else comparison.model_dump_json(indent=2)
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vc-clone-rehearsal")
    subparsers = parser.add_subparsers(dest="command", required=True)

    investors = subparsers.add_parser("investors")
    investors.add_argument("--config", type=Path, required=True)

    start = subparsers.add_parser("start")
    start.add_argument("--config", type=Path, required=True)
    start.add_argument("--vc", required=True)
    pitch = start.add_mutually_exclusive_group(required=True)
    pitch.add_argument("--pitch", type=Path)
    pitch.add_argument("--text")
    start.add_argument("--session")
    start.add_argument("--company", action="append")
    start.add_argument(
        "--rehearsal-depth",
        choices=("quick", "standard", "deep"),
        default="standard",
    )
    baseline = start.add_mutually_exclusive_group()
    baseline.add_argument("--build-canonical-baseline", action="store_true")
    baseline.add_argument("--canonical-baseline", type=Path)

    answer = subparsers.add_parser("answer")
    answer.add_argument("--config", type=Path, required=True)
    answer.add_argument("--session", required=True)
    answer.add_argument("--text", required=True)

    for name in ("finish", "retry", "verify"):
        command = subparsers.add_parser(name)
        command.add_argument("--config", type=Path, required=True)
        command.add_argument("--session", required=True)

    report = subparsers.add_parser("report")
    report.add_argument("--config", type=Path, required=True)
    report.add_argument("--session", required=True)
    report.add_argument("--format", choices=("json", "markdown"), default="json")

    compare = subparsers.add_parser("compare")
    compare.add_argument("--config", type=Path, required=True)
    compare.add_argument("--session", action="append", required=True)
    compare.add_argument("--format", choices=("json", "csv"), default="json")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        config = load_rehearsal_config(args.config)
        if args.command == "investors":
            command_investors(config)
        elif args.command == "start":
            command_start(config, args)
        elif args.command == "answer":
            _resume(config, args.session, "answer", args.text)
        elif args.command in {"finish", "retry"}:
            _resume(config, args.session, args.command)
        elif args.command == "report":
            command_report(config, args.session, args.format)
        elif args.command == "compare":
            command_compare(config, args.session, args.format)
        else:
            command_verify(config, args.session)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
