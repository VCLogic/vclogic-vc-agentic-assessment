"""Deterministic consolidation of Phase 1, Phase 2 and coverage outputs.

Rules come from the 21-22 October analysis of the 317-case benchmark:
- the salience/confidence-weighted direction tally of Phase 1 rationales ranks
  Ins better than the Phase 2 likelihood;
- Phase 1 is mostly positive, so the tally is only meaningful relative to the
  same investor's other pitches (per-VC cutoff);
- Phase 2 In verdicts that the tally does not support are mostly wrong (veto);
- averaging repeated runs reduces run-to-run noise.

On 83 later runs the Phase 1 tally alone did not hold up (AUC 0.48 versus
Phase 2 likelihood 0.60); adding coverage-expansion rationales restored it
(0.61). Over all 400 cases (runs and coverage averaged, leave-one-out cutoffs)
the tally with coverage gave the best balanced accuracy (0.573 versus 0.554 for
Phase 2) and ranking AUC (0.613 versus 0.582), so it is the default. On the 83
later cases alone, Phase 2 kept a higher balanced accuracy (0.585 versus 0.538)
and ranking was even (0.619 versus 0.615).

Calibration uses no outcome labels: cutoffs are quantiles of each investor's
historical tally distribution. Only the quota (default 30%) reflects the
benchmark's approximate In rate.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Iterable

TALLY_VERSION = "signed-salience-confidence-v1"
SIGN = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}
WEIGHT = {"primary": 1.0, "secondary": 0.5}


def rationale_score(item: dict[str, Any]) -> float:
    return SIGN.get(item.get("direction"), 0.0) * WEIGHT.get(item.get("salience"), 0.5) * float(
        item.get("confidence", 1.0)
    )


def tally(rationales: Iterable[dict[str, Any]]) -> float:
    return sum(rationale_score(r) for r in rationales)


def salience_order(rationales: list[dict[str, Any]], controlling: set[str]) -> list[dict[str, Any]]:
    return sorted(
        rationales,
        key=lambda r: (
            -(r.get("rationale_id") in controlling),
            -(r.get("salience") == "primary"),
            -float(r.get("confidence", 0.0)),
        ),
    )


def top_cutoff(values: list[float], quota: float) -> float:
    """Smallest score inside the top ``quota`` share of ``values``."""
    if not values:
        raise ValueError("cannot compute a cutoff without values")
    ranked = sorted(values, reverse=True)
    k = max(1, round(quota * len(ranked)))
    return ranked[k - 1]


def percentile(values: list[float], score: float) -> float:
    if not values:
        return float("nan")
    return (sum(v < score for v in values) + 0.5 * sum(v == score for v in values)) / len(values)


# --------------------------------------------------------------------------- history


def same_model(a: str | None, b: str | None) -> bool:
    """The Codex backend records OpenRouter models without the ``openai/`` prefix."""
    return (a or "").removeprefix("openai/") == (b or "").removeprefix("openai/")


def _coverage_values(value: float | list[float]) -> tuple[float, float]:
    """(mean over runs, first run) for a per-case coverage tally or list of per-run tallies."""
    values = value if isinstance(value, list) else [value]
    return sum(values) / len(values), values[0]


def historical_cases(
    evaluation: dict[str, Any], *, model: str | None = "openai/gpt-5.6-luna", input_mode: str = "founders-only"
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    """Group completed runs from an automated-evaluation report by VC/episode."""
    cases: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in evaluation.get("results", []):
        if row.get("execution_status") != "completed" or row.get("input_mode") != input_mode:
            continue
        if model is not None and not same_model(row.get("model"), model):
            continue
        phase1 = Path(row.get("phase1_path") or "")
        if not phase1.is_file():
            continue
        investigation = json.loads(phase1.read_text(encoding="utf-8"))
        cases[(row["vc_slug"], row["episode_slug"])].append({
            "tally": tally(investigation.get("rationales", [])),
            "likelihood": row.get("investment_likelihood"),
            "verdict": row.get("predicted_decision"),
            "actual": row.get("actual_pitch_decision"),
            "phase1_path": str(phase1),
        })
    return dict(cases)


def build_calibration(
    cases: dict[tuple[str, str], list[dict[str, Any]]],
    *,
    quota: float = 0.30,
    min_vc_cases: int = 8,
    coverage_tallies: dict[tuple[str, str], float | list[float]] | None = None,
) -> dict[str, Any]:
    """Cutoffs over historical tallies; with ``coverage_tallies`` each case adds its mean coverage tally."""
    if coverage_tallies is not None and (missing := set(cases) - set(coverage_tallies)):
        raise ValueError(f"{len(missing)} historical cases have no coverage output")
    per_vc: dict[str, list[float]] = defaultdict(list)
    for (vc, episode), runs in cases.items():
        extra = _coverage_values(coverage_tallies[(vc, episode)])[0] if coverage_tallies is not None else 0.0
        per_vc[vc].append(sum(r["tally"] for r in runs) / len(runs) + extra)
    pooled = [v for values in per_vc.values() for v in values]
    return {
        "schema": "components-calibration-v1",
        "tally_version": TALLY_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "quota": quota,
        "min_vc_cases": min_vc_cases,
        "uses_outcome_labels": False,
        "includes_coverage": coverage_tallies is not None,
        "global": {"n": len(pooled), "cutoff": top_cutoff(pooled, quota), "tallies": sorted(pooled)},
        "investors": {
            vc: {"n": len(values), "cutoff": top_cutoff(values, quota), "tallies": sorted(values)}
            for vc, values in sorted(per_vc.items())
        },
    }


def cutoff_for(
    calibration: dict[str, Any], vc: str, use_coverage: bool = False
) -> tuple[float, str, list[float]]:
    if calibration.get("tally_version") != TALLY_VERSION:
        raise ValueError("calibration was built with a different tally definition")
    if calibration.get("includes_coverage", False) != use_coverage:
        raise ValueError(
            "calibration and use_coverage disagree; rebuild with "
            "`vc-clone-components calibrate" + (" --coverage-root ...`" if use_coverage else "`")
        )
    entry = calibration["investors"].get(vc)
    if entry and entry["n"] >= calibration["min_vc_cases"]:
        return entry["cutoff"], "investor", entry["tallies"]
    return calibration["global"]["cutoff"], "global", calibration["global"]["tallies"]


# --------------------------------------------------------------------------- decision


def decide(rule: str, phase2_in: bool | None, tally_in: bool) -> bool:
    if rule == "tally":
        return tally_in
    if phase2_in is None:
        raise ValueError(f"rule {rule!r} requires a Phase 2 decision")
    if rule == "phase2":
        return phase2_in
    if rule == "veto":
        return phase2_in and tally_in
    raise ValueError(f"unknown rule: {rule}")


def _brief(item: dict[str, Any], source: str) -> dict[str, Any]:
    return {
        "source": source,
        "id": item.get("rationale_id") or item.get("addition_id"),
        "taxonomy_label": item.get("taxonomy_label"),
        "direction": item.get("direction"),
        "salience": item.get("salience"),
        "confidence": item.get("confidence"),
        "text": item.get("justification", ""),
    }


def consolidate(
    runs: list[Any],
    *,
    calibration: dict[str, Any],
    coverage: dict[str, Any] | list[dict[str, Any]] | None = None,
    rule: str = "tally",
    use_coverage: bool = False,
    coverage_weight: float = 1.0,
    quota: float | None = None,
) -> dict[str, Any]:
    """Combine one or more runs of the same VC/pitch with optional coverage output (one per run).

    ``quota`` overrides the calibration's share of In pitches for this investor: the cutoff is
    then the top ``quota`` of the same reference tallies.
    """
    if not runs:
        raise ValueError("at least one run is required")
    vc, episode = runs[0].vc_slug, runs[0].episode_slug
    if any((r.vc_slug, r.episode_slug) != (vc, episode) for r in runs):
        raise ValueError("all runs must assess the same investor and pitch")
    coverages = [c for c in (coverage if isinstance(coverage, list) else [coverage]) if c]
    if any((c.get("vc_slug"), c.get("episode_slug")) != (vc, episode) for c in coverages):
        raise ValueError("coverage output belongs to a different investor or pitch")
    usable = [c for c in coverages if c.get("status") in {"completed", "skipped_no_uncovered_labels"}]
    additions, seen = [], set()
    for item in (a for c in usable for a in c.get("additions", [])):
        if (item.get("taxonomy_label"), item.get("direction")) not in seen:
            seen.add((item.get("taxonomy_label"), item.get("direction")))
            additions.append(item)

    run_tallies = [tally(r.phase1.get("rationales", [])) for r in runs]
    base_tally = sum(run_tallies) / len(run_tallies)
    coverage_tallies = [tally(c.get("additions", [])) for c in usable]
    coverage_tally = coverage_weight * sum(coverage_tallies) / len(coverage_tallies) if usable else 0.0
    score = base_tally + (coverage_tally if use_coverage else 0.0)

    verdicts = [r.phase2.get("decision") for r in runs if r.phase2]
    likelihoods = [float(r.phase2["investment_likelihood"]) for r in runs if r.phase2]
    phase2_in = (sum(v == "In" for v in verdicts) > len(verdicts) / 2) if verdicts else None

    cutoff, cutoff_source, reference = cutoff_for(calibration, vc, use_coverage)
    if quota is not None:
        cutoff = top_cutoff(reference, quota)
    tally_in = score >= cutoff
    final_in = decide(rule, phase2_in, tally_in)

    primary = runs[0]
    controlling = set((primary.phase2 or {}).get("controlling_rationale_ids") or [])
    ordered = salience_order(primary.phase1.get("rationales", []), controlling)
    supporting = [_brief(r, "phase1") for r in ordered if r.get("direction") == "positive"][:3]
    opposing = [_brief(r, "phase1") for r in ordered if r.get("direction") == "negative"][:3]
    questions = [_brief(r, "phase1") for r in ordered if r.get("direction") == "neutral"][:3]

    flags = []
    if phase2_in is not None and phase2_in != tally_in:
        flags.append("phase2_tally_disagreement")
    if len(set(verdicts)) > 1:
        flags.append("run_verdicts_disagree")
    if cutoff_source == "global":
        flags.append("investor_history_too_small_global_cutoff_used")
    if additions and not use_coverage:
        flags.append("coverage_reported_not_scored")
    if use_coverage and not usable:
        flags.append("coverage_missing_tally_biased_low")
    elif use_coverage and len(usable) < len(runs):
        flags.append("coverage_missing_for_some_runs")
    if any(a.get("direction") == "negative" and a.get("salience") == "primary" for a in additions):
        flags.append("coverage_found_primary_negative")

    return {
        "schema": "consolidated-assessment-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "vc_slug": vc,
        "episode_slug": episode,
        "decision": "In" if final_in else "Out",
        "rule": rule,
        "scores": {
            "tally_version": TALLY_VERSION,
            "tally": score,
            "phase1_tally": base_tally,
            "phase1_tally_per_run": run_tallies,
            "coverage_tally": coverage_tally,
            "coverage_tally_per_run": coverage_tallies,
            "coverage_scored": use_coverage,
            "cutoff": cutoff,
            "cutoff_source": cutoff_source,
            "quota": calibration.get("quota") if quota is None else quota,
            "quota_source": "calibration" if quota is None else "investor_quota",
            "investor_percentile": percentile(reference, score),
            "tally_in": tally_in,
        },
        "phase2": {
            "decision": None if phase2_in is None else ("In" if phase2_in else "Out"),
            "run_decisions": verdicts,
            "mean_likelihood": sum(likelihoods) / len(likelihoods) if likelihoods else None,
        },
        "rationales": {
            "supporting": supporting,
            "opposing": opposing,
            "open_questions": questions,
            "coverage_additions": [_brief(a, "coverage") | {"pitch_quote": a.get("pitch_quote")} for a in additions],
        },
        "flags": flags,
        "inputs": {
            "run_dirs": [str(r.run_dir) for r in runs],
            "run_hashes": [r.hashes for r in runs],
            "coverage_status": [c.get("status") for c in coverages],
        },
        "note": "Automated development rules; validated only on retrospective Pitch Show runs.",
    }


# --------------------------------------------------------------------------- evaluation


def _auc(labels: list[int], scores: list[float]) -> float | None:
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    if not pos or not neg:
        return None
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


def _decision_metrics(labels: list[int], decisions: list[bool]) -> dict[str, Any]:
    tp = sum(y and d for y, d in zip(labels, decisions))
    tn = sum((not y) and (not d) for y, d in zip(labels, decisions))
    pos, neg = sum(labels), len(labels) - sum(labels)
    recall = tp / pos if pos else math.nan
    specificity = tn / neg if neg else math.nan
    predicted = sum(decisions)
    return {
        "accuracy": (tp + tn) / len(labels),
        "balanced_accuracy": (recall + specificity) / 2,
        "in_precision": tp / predicted if predicted else math.nan,
        "in_recall": recall,
        "predicted_ins": predicted,
    }


def evaluate_history(
    cases: dict[tuple[str, str], list[dict[str, Any]]],
    *,
    quota: float = 0.30,
    min_vc_cases: int = 8,
    coverage_tallies: dict[tuple[str, str], float | list[float]] | None = None,
) -> dict[str, Any]:
    """Leave-one-out evaluation: each case's cutoff excludes that case."""
    rows = []
    for key, runs in cases.items():
        actual = {r["actual"] for r in runs}
        if len(actual) != 1 or next(iter(actual)) not in {"In", "Out"}:
            continue
        verdicts = [r["verdict"] for r in runs]
        rows.append({
            "key": key,
            "y": int(next(iter(actual)) == "In"),
            "tally": sum(r["tally"] for r in runs) / len(runs),
            "first_tally": runs[0]["tally"],
            "likelihood": sum(float(r["likelihood"]) for r in runs) / len(runs),
            "phase2_in": sum(v == "In" for v in verdicts) > len(verdicts) / 2,
            "first_phase2_in": verdicts[0] == "In",
            "coverage": (coverage_tallies or {}).get(key),
        })
    by_vc: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_vc[row["key"][0]].append(row)

    def loo_cutoff(row: dict[str, Any], field: str) -> float:
        own = [r[field] for r in by_vc[row["key"][0]] if r is not row]
        if len(own) >= min_vc_cases:
            return top_cutoff(own, quota)
        return top_cutoff([r[field] for r in rows if r is not row], quota)

    labels = [r["y"] for r in rows]
    report: dict[str, Any] = {
        "cases": len(rows),
        "actual_ins": sum(labels),
        "quota": quota,
        "always_out": _decision_metrics(labels, [False] * len(rows)),
        "ranking_auc": {
            "phase2_likelihood": _auc(labels, [r["likelihood"] for r in rows]),
            "tally_mean_of_runs": _auc(labels, [r["tally"] for r in rows]),
            "tally_first_run": _auc(labels, [r["first_tally"] for r in rows]),
        },
        "decisions": {},
    }
    for name, field, phase2_field in (
        ("first_run", "first_tally", "first_phase2_in"),
        ("mean_of_runs", "tally", "phase2_in"),
    ):
        tally_in = [r[field] >= loo_cutoff(r, field) for r in rows]
        report["decisions"][f"phase2_{name}"] = _decision_metrics(labels, [r[phase2_field] for r in rows])
        report["decisions"][f"tally_{name}"] = _decision_metrics(labels, tally_in)
        report["decisions"][f"veto_{name}"] = _decision_metrics(
            labels, [r[phase2_field] and t for r, t in zip(rows, tally_in)]
        )
    covered = [r for r in rows if r["coverage"] is not None]
    if covered:
        for r in covered:
            mean_coverage, first_coverage = _coverage_values(r["coverage"])
            r["tally_cov"] = r["tally"] + mean_coverage
            r["first_tally_cov"] = r["first_tally"] + first_coverage
        sub_labels = [r["y"] for r in covered]
        subset: dict[str, Any] = {
            "cases": len(covered),
            "phase2_likelihood_auc": _auc(sub_labels, [r["likelihood"] for r in covered]),
            "tally_auc": _auc(sub_labels, [r["tally"] for r in covered]),
            "tally_with_coverage_auc": _auc(sub_labels, [r["tally_cov"] for r in covered]),
            "decisions_with_coverage": {},
        }
        if len(covered) == len(rows):  # cutoffs need every case's coverage tally
            for name, field, phase2_field in (
                ("first_run", "first_tally_cov", "first_phase2_in"),
                ("mean_of_runs", "tally_cov", "phase2_in"),
            ):
                tally_in = [r[field] >= loo_cutoff(r, field) for r in covered]
                subset["decisions_with_coverage"][f"tally_{name}"] = _decision_metrics(sub_labels, tally_in)
                subset["decisions_with_coverage"][f"veto_{name}"] = _decision_metrics(
                    sub_labels, [r[phase2_field] and t for r, t in zip(covered, tally_in)]
                )
        report["coverage_subset"] = subset
    return report
