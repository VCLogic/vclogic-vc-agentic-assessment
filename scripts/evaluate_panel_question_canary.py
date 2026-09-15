#!/usr/bin/env python3
"""Paired rationale evaluation of panel-question and founder-only Phase 1 inputs."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import json
from pathlib import Path
import random
from statistics import fmean

from vc_clone_graph.phase1_evaluation import (
    Phase1Case,
    _prediction_rationales,
    evaluate_cases,
    load_phase1_cases,
)


DEFAULT_REGISTRY = Path("evaluation/canonical_runs_v4_v41_portfolio_2026-08-15.json")
DEFAULT_REFERENCES = Path("evaluation/phase1_ground_truth_rationales")
DEFAULT_TAXONOMY = Path("../agentic-vc-clone-framework/taxonomy/codebook_v_final.json")


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _condition_summary(result: dict) -> dict:
    weighted = {
        (row["scope"], row["mode"]): row
        for row in result["weighted_recovery_rows"]
    }
    overall = result["overall"]
    salience = weighted[("overall", "salience")]
    return {
        "n": result["case_count"],
        "micro_precision": overall["micro_precision"],
        "micro_recall": overall["micro_recall"],
        "micro_f1": overall["micro_f1"],
        "macro_jaccard": overall["macro_jaccard"],
        "exact_set_match_rate": overall["exact_set_match_rate"],
        "at_least_one_correct_rate": overall["at_least_one_correct_rate"],
        "average_predicted_size": overall["average_predicted_size"],
        "salience_weighted_precision": salience["precision"],
        "salience_weighted_recall": salience["recall"],
        "salience_weighted_f1": salience["f1"],
    }


def _vc_summaries(result: dict) -> dict[str, dict]:
    return {
        row["value"]: {
            key: row[key]
            for key in (
                "n", "micro_precision", "micro_recall", "micro_f1",
                "macro_jaccard", "at_least_one_correct_rate", "average_predicted_size",
            )
        }
        for row in result["scope_rows"]
        if row["dimension"] == "vc"
    }


def _paired_uncertainty(deltas: list[float], *, seed: int = 20260820) -> dict:
    """Deterministic paired bootstrap CI and sign-randomization diagnostic."""
    if not deltas:
        raise ValueError("paired uncertainty requires episode deltas")
    rng = random.Random(seed)
    bootstrap = sorted(
        fmean(rng.choice(deltas) for _ in deltas)
        for _ in range(20_000)
    )
    observed = fmean(deltas)
    exceedances = 0
    permutations = 100_000
    for _ in range(permutations):
        randomized = fmean(value if rng.random() < 0.5 else -value for value in deltas)
        exceedances += abs(randomized) >= abs(observed)
    return {
        "estimand": "mean paired episode-level exact-label F1 delta",
        "mean_delta": observed,
        "bootstrap_95_ci": [bootstrap[499], bootstrap[19_499]],
        "sign_randomization_p_two_sided": (exceedances + 1) / (permutations + 1),
        "bootstrap_samples": 20_000,
        "randomization_samples": permutations,
        "seed": seed,
    }


def _economics(experiment_root: Path, status: dict) -> dict:
    selected_cost = sum(
        float(attempt["summary"]["usage"].get("cost_usd", 0) or 0)
        for case in status["cases"].values()
        for attempt in case["attempts"]
        if attempt.get("status") == "completed"
    )
    all_attempt_cost = 0.0
    paid_attempt_count = 0
    for attempt_root in sorted((experiment_root / "runs").glob("attempt-*")):
        for episode_root in attempt_root.glob("*/*"):
            summary_path = episode_root / "summary.json"
            if summary_path.is_file():
                cost = float(
                    json.loads(summary_path.read_text(encoding="utf-8"))
                    .get("usage", {})
                    .get("cost_usd", 0) or 0
                )
            else:
                cost = 0.0
                for response in episode_root.rglob("*model-response*.json"):
                    try:
                        cost += float(
                            json.loads(response.read_text(encoding="utf-8"))
                            .get("usage", {})
                            .get("cost_usd", 0) or 0
                        )
                    except (TypeError, ValueError, json.JSONDecodeError):
                        continue
            if cost:
                paid_attempt_count += 1
                all_attempt_cost += cost
    cases = list(status["cases"].values())
    retried = sum(len(case["attempts"]) > 1 for case in cases)
    return {
        "case_count": len(cases),
        "first_attempt_completion_rate": (len(cases) - retried) / len(cases),
        "final_completion_rate": sum(case["status"] == "completed" for case in cases) / len(cases),
        "retried_case_count": retried,
        "paid_attempt_count": paid_attempt_count,
        "selected_output_cost_usd": selected_cost,
        "total_observed_cost_usd": all_attempt_cost,
        "retry_or_interrupted_cost_usd": all_attempt_cost - selected_cost,
        "mean_total_cost_per_case_usd": all_attempt_cost / len(cases),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, default=Path("outputs/panel-question-phase1-canary-2026-08-20"))
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--references", type=Path, default=DEFAULT_REFERENCES)
    parser.add_argument("--taxonomy", type=Path, default=DEFAULT_TAXONOMY)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()

    selection = json.loads((args.experiment_root / "selection.json").read_text(encoding="utf-8"))
    status = json.loads((args.experiment_root / "batch-status.json").read_text(encoding="utf-8"))
    canonical, taxonomy = load_phase1_cases(
        Path.cwd(), args.registry, args.references, args.taxonomy
    )
    canonical_by_key = {(row.vc_slug, row.episode_slug): row for row in canonical}

    founder_only: list[Phase1Case] = []
    panel_question: list[Phase1Case] = []
    skipped: list[str] = []
    for selected in selection["cases"]:
        batch_key = f"{selected['vc_slug']}::{selected['episode_slug']}"
        batch = status["cases"].get(batch_key)
        if not batch or batch.get("status") != "completed":
            skipped.append(batch_key)
            continue
        attempt = next(
            row for row in reversed(batch["attempts"])
            if row.get("status") == "completed"
        )
        artifact = Path(attempt["artifact_root"]) / "phase1/investigation.json"
        payload = json.loads(artifact.read_text(encoding="utf-8"))
        registry_key = (selected["registry_vc_slug"], selected["episode_slug"])
        source = canonical_by_key[registry_key]
        founder_only.append(source)
        panel_question.append(replace(
            source,
            predicted=_prediction_rationales(payload, taxonomy, str(artifact)),
            artifact_path=artifact,
        ))

    if skipped and not args.allow_incomplete:
        raise ValueError(f"experiment has {len(skipped)} incomplete cases: {skipped}")
    if not panel_question:
        raise ValueError("no completed treatment cases")

    control_result = evaluate_cases(founder_only, taxonomy)
    treatment_result = evaluate_cases(panel_question, taxonomy)
    control = _condition_summary(control_result)
    treatment = _condition_summary(treatment_result)
    control_vc = _vc_summaries(control_result)
    treatment_vc = _vc_summaries(treatment_result)

    episode_control = {
        (row["vc_slug"], row["episode_slug"]): row
        for row in control_result["episode_rows"]
    }
    episode_treatment = {
        (row["vc_slug"], row["episode_slug"]): row
        for row in treatment_result["episode_rows"]
    }
    episode_rows = []
    for key in sorted(episode_treatment):
        before, after = episode_control[key], episode_treatment[key]
        episode_rows.append({
            "vc_slug": key[0],
            "episode_slug": key[1],
            "actual_decision": after["actual_decision"],
            "control_precision": before["precision"],
            "control_recall": before["recall"],
            "control_f1": before["f1"],
            "treatment_precision": after["precision"],
            "treatment_recall": after["recall"],
            "treatment_f1": after["f1"],
            "delta_f1": after["f1"] - before["f1"],
            "control_labels": before["top_labels"],
            "treatment_labels": after["top_labels"],
        })

    newly_recovered: Counter[str] = Counter()
    newly_lost: Counter[str] = Counter()
    for control_case, treatment_case in zip(founder_only, panel_question, strict=True):
        reference = {row.label for row in control_case.reference}
        before = {row.label for row in control_case.predicted}
        after = {row.label for row in treatment_case.predicted}
        newly_recovered.update((after - before) & reference)
        newly_lost.update((before - after) & reference)

    result = {
        "schema": "panel-question-phase1-comparison-v1",
        "completed_cases": len(panel_question),
        "skipped_cases": skipped,
        "control": control,
        "treatment": treatment,
        "delta": {key: treatment[key] - control[key] for key in control if key != "n"},
        "per_vc": {
            vc: {
                "control": control_vc[vc],
                "treatment": treatment_vc[vc],
                "delta_micro_f1": treatment_vc[vc]["micro_f1"] - control_vc[vc]["micro_f1"],
            }
            for vc in sorted(treatment_vc)
        },
        "episode_rows": episode_rows,
        "improved_episode_rate": fmean(row["delta_f1"] > 0 for row in episode_rows),
        "worsened_episode_rate": fmean(row["delta_f1"] < 0 for row in episode_rows),
        "unchanged_episode_rate": fmean(row["delta_f1"] == 0 for row in episode_rows),
        "paired_uncertainty": _paired_uncertainty([row["delta_f1"] for row in episode_rows]),
        "economics": _economics(args.experiment_root, status),
        "rationale_shifts": {
            "newly_recovered": dict(newly_recovered.most_common()),
            "formerly_recovered_now_lost": dict(newly_lost.most_common()),
        },
    }
    output = args.experiment_root / "evaluation"
    _write(output / "comparison.json", json.dumps(result, indent=2, sort_keys=True) + "\n")

    lines = [
        "# Panel-question Phase 1 comparison",
        "",
        f"Paired sample: **{len(panel_question)}** cases; incomplete: **{len(skipped)}**.",
        "",
        "| Condition | Precision | Recall | F1 | Jaccard | Salience-weighted F1 | Avg rationales |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, row in (("Founder-only", control), ("Panel questions + founder answers", treatment)):
        lines.append(
            f"| {label} | {row['micro_precision']:.3f} | {row['micro_recall']:.3f} | "
            f"{row['micro_f1']:.3f} | {row['macro_jaccard']:.3f} | "
            f"{row['salience_weighted_f1']:.3f} | {row['average_predicted_size']:.2f} |"
        )
    lines.extend(["", "## Per VC", "", "| VC | Control F1 | Treatment F1 | Delta |", "|---|---:|---:|---:|"])
    for vc, row in result["per_vc"].items():
        lines.append(f"| {vc} | {row['control']['micro_f1']:.3f} | {row['treatment']['micro_f1']:.3f} | {row['delta_micro_f1']:+.3f} |")
    uncertainty = result["paired_uncertainty"]
    lines.extend([
        "",
        "## Paired uncertainty",
        "",
        f"Mean episode-level F1 change: **{uncertainty['mean_delta']:+.3f}**; paired bootstrap 95% CI "
        f"**[{uncertainty['bootstrap_95_ci'][0]:+.3f}, {uncertainty['bootstrap_95_ci'][1]:+.3f}]**; "
        f"sign-randomization p = **{uncertainty['sign_randomization_p_two_sided']:.3f}**. "
        f"Improved/unchanged/worsened episodes: "
        f"**{sum(row['delta_f1'] > 0 for row in episode_rows)}/"
        f"{sum(row['delta_f1'] == 0 for row in episode_rows)}/"
        f"{sum(row['delta_f1'] < 0 for row in episode_rows)}**.",
    ])
    lines.extend([
        "",
        "## Interpretation guardrail",
        "",
        "This is a paired 24-case development experiment, selected for safe question availability rather than random sampling. It estimates whether restoring panel-question framing changes rationale recovery; it is not an untouched holdout estimate.",
    ])
    economics = result["economics"]
    lines.extend([
        "",
        "## Reliability and economics",
        "",
        f"First-attempt completion was **{economics['first_attempt_completion_rate']:.1%}**; "
        f"final completion after one retry was **{economics['final_completion_rate']:.1%}**. "
        f"Total observed OpenRouter cost was **${economics['total_observed_cost_usd']:.3f}** "
        f"(**${economics['mean_total_cost_per_case_usd']:.3f} per case**), including "
        f"**${economics['retry_or_interrupted_cost_usd']:.3f}** in failed or interrupted attempts.",
        "",
        "## Rationale shifts",
        "",
        "Restored questions changed which labels were activated rather than simply adding coverage. "
        f"Most frequently newly recovered: "
        f"{', '.join(f'{label} ({count})' for label, count in newly_recovered.most_common(5)) or 'none'}. "
        f"Most frequently formerly recovered but now lost: "
        f"{', '.join(f'{label} ({count})' for label, count in newly_lost.most_common(5)) or 'none'}.",
    ])
    _write(output / "REPORT.md", "\n".join(lines) + "\n")
    print(json.dumps({"completed": len(panel_question), "skipped": len(skipped), "control_f1": control["micro_f1"], "treatment_f1": treatment["micro_f1"], "delta_f1": result["delta"]["micro_f1"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
