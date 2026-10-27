"""Command line for the reusable post-assessment components."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
from typing import Any

from .config import ComponentsConfig, load_components_config
from .consolidate import (
    build_calibration,
    consolidate,
    evaluate_history,
    historical_cases,
    tally,
)
from .coverage import build_prompt, dumps, openrouter_provider, run_coverage
from .factsheet import run_factsheet
from .lens import _run_package, run_lens
from .run_inputs import load_run

COMPONENTS_DIR = "components"


def _read(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _read_evaluation(path: Path) -> dict[str, Any]:
    """Read an automated-evaluation report, following the ``latest.json`` pointer."""
    value = _read(path)
    if "results" not in value and "report_path" in value:
        value = _read(Path(value["report_path"]))
    if "results" not in value:
        raise ValueError(f"not an automated-evaluation report: {path}")
    return value


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dumps(value), encoding="utf-8")


def _config(path: Path | None) -> ComponentsConfig:
    return load_components_config(path) if path else ComponentsConfig()


def run_pipeline_components(
    run_dir: Path, components_path: Path | None, workspace: Path = Path(".")
) -> dict[str, Any]:
    """Hook for ``vc-clone-graph run``; writes into ``<run>/components/``.

    ``components_path=None`` uses the built-in defaults.
    """
    config = load_components_config(components_path) if components_path is not None else ComponentsConfig()
    out = Path(run_dir) / COMPONENTS_DIR
    summary: dict[str, Any] = {}
    coverage = None
    if config.coverage.enabled:
        run = load_run(run_dir, workspace=workspace)
        coverage = run_coverage(run, config.coverage, openrouter_provider(config.coverage))
        _write(out / "coverage.json", coverage)
        summary["coverage_additions"] = len(coverage["additions"])
    if config.consolidate.enabled:
        run = load_run(run_dir, workspace=workspace, require_phase2=True)
        settings = config.consolidate
        result = consolidate(
            [run],
            calibration=_read(workspace / settings.calibration_path),
            coverage=coverage,
            rule=settings.rule,
            use_coverage=settings.use_coverage,
            coverage_weight=settings.coverage_weight,
            quota=settings.investor_quota.get(run.vc_slug),
        )
        _write(out / "consolidated.json", result)
        summary["consolidated_decision"] = result["decision"]
    if config.lens.enabled:
        # Runs after consolidate so a lens failure never blocks the decision. A single run
        # has no panel, so the contrastive lens uses the investor's own profile only.
        run = load_run(run_dir, workspace=workspace)
        lens = run_lens(run, config.lens, openrouter_provider(config.lens), coverage=coverage, workspace=workspace)
        _write(out / "lens.json", lens)
        summary["lens_rationales"] = len(lens["rationales"])
    return summary


def _backfill_path(root: Path, vc: str, episode: str, phase1_path: str) -> Path:
    key = sha256(phase1_path.encode()).hexdigest()[:16]
    return root / vc / episode / key / "coverage.json"


def _coverage_tallies(evaluation: dict[str, Any], root: Path, model: str | None) -> dict[tuple[str, str], list[float]]:
    """Per-run coverage tallies, for cases where every run has usable coverage."""
    found: dict[tuple[str, str], list[float]] = {}
    for key, runs in historical_cases(evaluation, model=model).items():
        values = []
        for run in runs:
            path = _backfill_path(root, key[0], key[1], run["phase1_path"])
            data = _read(path) if path.is_file() else {}
            if data.get("status") not in {"completed", "skipped_no_uncovered_labels"}:
                break
            values.append(tally(data.get("additions", [])))
        else:
            found[key] = values
    return found


def _latest_runs(evaluation: dict[str, Any], model: str | None) -> dict[tuple[str, str], dict[str, Any]]:
    """The most recent run of each pitch, by Phase 2 output time."""
    finished = {
        str(Path(row["phase1_path"])): Path(row["phase2_path"]).stat().st_mtime
        for row in evaluation.get("results", [])
        if row.get("phase1_path") and row.get("phase2_path") and Path(row["phase2_path"]).is_file()
    }
    cases = historical_cases(evaluation, model=model)
    return {
        key: max(runs, key=lambda r: finished.get(r["phase1_path"], 0.0)) for key, runs in cases.items()
    }


def _episode(episodes_root: Path, slug: str) -> dict[str, Any]:
    for path in sorted(episodes_root.glob(f"*/source/pitch-show/episodes/{slug}.json")):
        return _read(path)
    raise FileNotFoundError(f"no episode transcript for {slug}")


def _factsheet_path(root: Path, slug: str) -> Path:
    return root / f"{slug}.json"


def _lens_path(root: Path, variant: str, vc: str, episode: str) -> Path:
    return root / variant / vc / episode / "lens.json"


def _lens_label(settings) -> str:
    label = f"{settings.variant}-keep{settings.max_kept}" if settings.max_kept else settings.variant
    return label if settings.include_memory else f"{label}-nomemory"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="vc-clone-components")
    sub = parser.add_subparsers(dest="command", required=True)

    cov = sub.add_parser("coverage", help="Expand Phase 1 coverage for one run (LLM call)")
    cov.add_argument("--run-dir", type=Path, required=True)
    cov.add_argument("--config", type=Path)
    cov.add_argument("--taxonomy", type=Path)
    cov.add_argument("--out", type=Path)
    cov.add_argument("--dry-run", action="store_true", help="Print the prompt size; make no call")

    back = sub.add_parser("backfill-coverage", help="Run coverage over completed historical runs")
    back.add_argument("--evaluation", type=Path, default=Path("evaluation/pitchshow/results/automated-evaluation/latest.json"))
    back.add_argument("--config", type=Path)
    back.add_argument("--out-root", type=Path, default=Path("outputs/components-backfill"))
    back.add_argument("--vc", action="append", default=[])
    back.add_argument("--limit", type=int)
    back.add_argument("--execute", action="store_true", help="Authorize paid coverage calls")

    cal = sub.add_parser("calibrate", help="Build per-investor tally cutoffs from past runs")
    cal.add_argument("--evaluation", type=Path, default=Path("evaluation/pitchshow/results/automated-evaluation/latest.json"))
    cal.add_argument("--out", type=Path, required=True)
    cal.add_argument("--quota", type=float, default=0.30)
    cal.add_argument("--min-vc-cases", type=int, default=8)
    cal.add_argument("--coverage-root", type=Path, help="Backfilled coverage; required when use_coverage = true")

    con = sub.add_parser("consolidate", help="Combine Phase 1, Phase 2 and coverage for one pitch")
    con.add_argument("--run-dir", type=Path, action="append", required=True, help="Repeat for repeated runs")
    con.add_argument("--coverage", type=Path, action="append", help="Repeat, one per --run-dir")
    con.add_argument("--config", type=Path)
    con.add_argument("--calibration", type=Path)
    con.add_argument("--rule", choices=("veto", "tally", "phase2"))
    con.add_argument("--use-coverage", action="store_true")
    con.add_argument("--out", type=Path)

    ev = sub.add_parser("evaluate", help="Leave-one-out evaluation of consolidation rules on past runs")
    ev.add_argument("--evaluation", type=Path, default=Path("evaluation/pitchshow/results/automated-evaluation/latest.json"))
    ev.add_argument("--coverage-root", type=Path)
    ev.add_argument("--quota", type=float, default=0.30)
    ev.add_argument("--min-vc-cases", type=int, default=8)
    ev.add_argument("--out", type=Path)

    fs = sub.add_parser("factsheet", help="Founder in-room fact sheets, one per episode (LLM call)")
    fs.add_argument("--episode", action="append", default=[], help="Repeat; default: every evaluated episode")
    fs.add_argument("--evaluation", type=Path, default=Path("evaluation/pitchshow/results/automated-evaluation/latest.json"))
    fs.add_argument("--reviews", type=Path, default=Path("evaluation/pitchshow/reviews"))
    fs.add_argument("--episodes-root", type=Path, default=Path("onboarding"))
    fs.add_argument("--config", type=Path)
    fs.add_argument("--out-root", type=Path)
    fs.add_argument("--execute", action="store_true", help="Authorize paid calls")

    ln = sub.add_parser("lens", help="Investor lens on each pitch's most recent run (LLM call)")
    ln.add_argument("--variant", choices=("memory", "precedents", "facts", "profile", "contrastive", "fit", "unique-wiki", "unique-record"))
    ln.add_argument("--no-memory", action="store_true", help="Leave the Investment Memory out of the lens")
    ln.add_argument("--max-kept", type=int, help="Selective lens: keep at most this many reasons")
    ln.add_argument("--episode", action="append", default=[])
    ln.add_argument("--vc", action="append", default=[])
    ln.add_argument("--evaluation", type=Path, default=Path("evaluation/pitchshow/results/automated-evaluation/latest.json"))
    ln.add_argument("--coverage-root", type=Path, default=Path("outputs/components-backfill"))
    ln.add_argument("--config", type=Path)
    ln.add_argument("--out-root", type=Path, default=Path("outputs/components-lens"))
    ln.add_argument("--limit", type=int)
    ln.add_argument("--execute", action="store_true", help="Authorize paid calls")

    un = sub.add_parser("uniqueness", help="Write each investor's uniqueness page (LLM call per investor)")
    un.add_argument("--vc", action="append", default=[])
    un.add_argument("--episode", action="append", default=[])
    un.add_argument("--evaluation", type=Path, default=Path("evaluation/pitchshow/results/automated-evaluation/latest.json"))
    un.add_argument("--aliases", type=Path, default=Path("evaluation/pitchshow/company-aliases.json"))
    un.add_argument("--wiki-root", type=Path, default=Path("inputs/wiki"))
    un.add_argument("--episodes-root", type=Path, default=Path("onboarding"))
    un.add_argument("--config", type=Path)
    un.add_argument("--execute", action="store_true", help="Authorize paid calls")

    for p in (back, cal, ev, fs, ln, un):
        p.add_argument("--model", default="openai/gpt-5.6-luna", help="Filter runs by model; 'any' disables")

    args = parser.parse_args(argv)
    try:
        if args.command == "coverage":
            config = _config(args.config).coverage
            run = load_run(args.run_dir, taxonomy_path=args.taxonomy)
            if args.dry_run:
                prompt, labels = build_prompt(run, config)
                print(json.dumps({"prompt_chars": len(prompt), "approx_tokens": len(prompt) // 4,
                                  "uncovered_labels": len(labels)}))
                return 0
            result = run_coverage(run, config, openrouter_provider(config))
            _write(args.out or args.run_dir / COMPONENTS_DIR / "coverage.json", result)
            print(json.dumps({"status": result["status"], "additions": len(result["additions"]),
                              "rejected": len(result["rejected"])}))

        elif args.command == "backfill-coverage":
            config = _config(args.config).coverage
            model = None if args.model == "any" else args.model
            cases = historical_cases(_read_evaluation(args.evaluation), model=model)
            todo = []
            for (vc, episode), runs in sorted(cases.items()):
                if args.vc and vc not in args.vc:
                    continue
                for run in runs:
                    path = _backfill_path(args.out_root, vc, episode, run["phase1_path"])
                    if not path.is_file():
                        todo.append((Path(run["phase1_path"]).parent.parent, path))
            todo = todo[: args.limit] if args.limit else todo
            if not args.execute:
                chars = [len(build_prompt(load_run(d), config)[0]) for d, _ in todo[:20]]
                print(json.dumps({"pending": len(todo), "sampled": len(chars),
                                  "mean_approx_input_tokens": (sum(chars) // max(1, len(chars))) // 4,
                                  "note": "dry run; pass --execute to make paid calls"}))
                return 0
            provider = openrouter_provider(config)
            done = failed = 0
            for run_dir, path in todo:
                try:
                    _write(path, run_coverage(load_run(run_dir), config, provider))
                    done += 1
                except Exception as exc:  # keep going; record the failure
                    _write(path.with_name("coverage-error.json"), {"run_dir": str(run_dir), "error": f"{type(exc).__name__}: {exc}"})
                    failed += 1
            print(json.dumps({"completed": done, "failed": failed}))

        elif args.command == "factsheet":
            settings = _config(args.config).factsheet
            lens_settings = _config(args.config).lens
            out_root = args.out_root or Path(lens_settings.factsheet_root)
            model = None if args.model == "any" else args.model
            slugs = args.episode or sorted({ep for _, ep in _latest_runs(_read_evaluation(args.evaluation), model)})
            todo = [s for s in slugs if not _factsheet_path(out_root, s).is_file()]
            if not args.execute:
                print(json.dumps({"pending": len(todo), "note": "dry run; pass --execute to make paid calls"}))
                return 0
            from ..dataset_prep import parse_speaker_turns
            provider = openrouter_provider(settings)
            done = failed = 0
            for slug in todo:
                try:
                    episode = _episode(args.episodes_root, slug)
                    turns = parse_speaker_turns(episode.get("transcript") or "")
                    panel = [p["name"] for p in episode.get("panel", []) if p.get("name")]
                    review = _read(args.reviews / f"{slug}.json")
                    _write(_factsheet_path(out_root, slug), run_factsheet(review, turns, panel, settings, provider))
                    done += 1
                except Exception as exc:  # keep going; record the failure
                    _write(out_root / "errors" / f"{slug}.json", {"episode": slug, "error": f"{type(exc).__name__}: {exc}"})
                    failed += 1
            print(json.dumps({"completed": done, "failed": failed}))

        elif args.command == "lens":
            config = _config(args.config)
            update = {k: v for k, v in (("variant", args.variant), ("max_kept", args.max_kept)) if v is not None}
            if args.no_memory:
                update["include_memory"] = False
            settings = config.lens.model_validate({**config.lens.model_dump(), **update})
            model = None if args.model == "any" else args.model
            evaluation = _read_evaluation(args.evaluation)
            todo = []
            latest = _latest_runs(evaluation, model)
            for (vc, episode), run in sorted(latest.items()):
                if (args.vc and vc not in args.vc) or (args.episode and episode not in args.episode):
                    continue
                path = _lens_path(args.out_root, _lens_label(settings), vc, episode)
                if not path.is_file():
                    todo.append((vc, episode, run, path))
            todo = todo[: args.limit] if args.limit else todo
            if not args.execute:
                print(json.dumps({"pending": len(todo), "variant": _lens_label(settings),
                                  "note": "dry run; pass --execute to make paid calls"}))
                return 0
            provider = openrouter_provider(settings)
            done = failed = 0
            for vc, episode, row, path in todo:
                try:
                    run = load_run(Path(row["phase1_path"]).parent.parent)
                    coverage_path = _backfill_path(args.coverage_root, vc, episode, row["phase1_path"])
                    coverage = _read(coverage_path) if coverage_path.is_file() else None
                    facts_path = _factsheet_path(Path(settings.factsheet_root), episode)
                    facts = _read(facts_path) if facts_path.is_file() else None
                    panel = [
                        (other, load_run(Path(r["phase1_path"]).parent.parent).investor_name)
                        for (other, ep), r in sorted(latest.items()) if ep == episode and other != vc
                    ] if settings.variant == "contrastive" else []
                    result = run_lens(run, settings, provider, coverage=coverage, facts=facts, panel=panel)
                    result["phase1_path"] = row["phase1_path"]
                    _write(path, result)
                    done += 1
                except Exception as exc:  # keep going; record the failure
                    _write(path.with_name("lens-error.json"), {"phase1_path": row["phase1_path"], "error": f"{type(exc).__name__}: {exc}"})
                    failed += 1
            print(json.dumps({"completed": done, "failed": failed}))

        elif args.command == "uniqueness":
            from .uniqueness import build_page, render, wiki_statistics

            settings = _config(args.config).lens
            out_root = Path(settings.uniqueness_root)
            latest = _latest_runs(_read_evaluation(args.evaluation), None if args.model == "any" else args.model)
            todo = [(vc, episode, row) for (vc, episode), row in sorted(latest.items())
                    if (not args.vc or vc in args.vc) and (not args.episode or episode in args.episode)
                    and not (out_root / vc / f"{episode}.json").is_file()]
            if not args.execute:
                print(json.dumps({"pending": len(todo), "note": "dry run; pass --execute to make paid calls"}))
                return 0
            stats = wiki_statistics(args.wiki_root)
            provider = openrouter_provider(settings)
            done = failed = 0
            for vc, episode, row in todo:
                try:
                    run = load_run(Path(row["phase1_path"]).parent.parent)
                    _, package = _run_package(run, Path("."))
                    page = build_page(run, settings, provider, stats=stats, aliases=package.target_company_aliases)
                    _write(out_root / vc / f"{episode}.json", page)
                    (out_root / vc / f"{episode}.md").write_text(render(page) + "\n", encoding="utf-8")
                    done += 1
                except Exception as exc:  # keep going; record the failure
                    _write(out_root / vc / f"{episode}-error.json", {"error": f"{type(exc).__name__}: {exc}"})
                    failed += 1
            print(json.dumps({"completed": done, "failed": failed}))

        elif args.command == "calibrate":
            model = None if args.model == "any" else args.model
            cases = historical_cases(_read_evaluation(args.evaluation), model=model)
            coverage = None
            if args.coverage_root is not None:
                coverage = _coverage_tallies(_read_evaluation(args.evaluation), args.coverage_root, model)
            calibration = build_calibration(
                cases, quota=args.quota, min_vc_cases=args.min_vc_cases, coverage_tallies=coverage
            )
            calibration["source"] = str(args.evaluation)
            _write(args.out, calibration)
            print(json.dumps({"cases": calibration["global"]["n"], "investors": len(calibration["investors"])}))

        elif args.command == "consolidate":
            settings = _config(args.config).consolidate
            calibration_path = args.calibration or Path(settings.calibration_path)
            runs = [load_run(d, require_phase2=args.rule != "tally") for d in args.run_dir]
            coverage_paths = args.coverage or [d / COMPONENTS_DIR / "coverage.json" for d in args.run_dir]
            result = consolidate(
                runs,
                calibration=_read(calibration_path),
                coverage=[_read(p) for p in coverage_paths if p.is_file()],
                rule=args.rule or settings.rule,
                use_coverage=args.use_coverage or settings.use_coverage,
                coverage_weight=settings.coverage_weight,
                quota=settings.investor_quota.get(runs[0].vc_slug),
            )
            _write(args.out or args.run_dir[0] / COMPONENTS_DIR / "consolidated.json", result)
            print(json.dumps({"decision": result["decision"], "flags": result["flags"]}))

        else:
            model = None if args.model == "any" else args.model
            evaluation = _read_evaluation(args.evaluation)
            cases = historical_cases(evaluation, model=model)
            coverage = _coverage_tallies(evaluation, args.coverage_root, model) if args.coverage_root else None
            report = evaluate_history(cases, quota=args.quota, min_vc_cases=args.min_vc_cases, coverage_tallies=coverage)
            if args.out:
                _write(args.out, report)
            print(json.dumps(report, indent=2))
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
