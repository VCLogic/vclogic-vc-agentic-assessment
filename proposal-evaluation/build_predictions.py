"""Write predictions.csv, one row per rationale the twin gave, from assessment runs.

    uv run python proposal-evaluation/build_predictions.py --runs research/predictions_original \
        --out proposal-evaluation/data/predictions.csv

Each --runs folder is searched for two layouts:

- a case folder (the data bundle's research/predictions_original/<investor>/<episode>/) with
  investigation.json, decision.json, and optionally coverage.json and final_lens.json;
- a live run folder (outputs/.../runs/<episode>/) with run-config.json, phase1/investigation.json,
  phase2/decision.json, and optionally coverage.json and lens.json, either beside them or in
  components/.

Rows by stage: investigator (Phase 1 rationales), taxonomy_review (coverage additions) and
decision_agent (final lens rationales). Every row of a decision repeats its weighted rationale
score and call, and the decision agent's call and likelihood. The weighted score sums direction
(+1 / -1 / 0) x salience (1 primary, 0.5 secondary) x confidence over the investigator and
taxonomy-review rationales; the call is In when the score is in the investor's top 30% of their
other pitches (all other pitches when the investor has fewer than eight), as in evaluate.py.
When one pitch has several runs, the most recent decision is kept.
"""
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

SIGN = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}
WEIGHT = {"primary": 1.0, "secondary": 0.5}
QUOTA = 0.3
MIN_OWN = 8
STAGES = ("investigator", "taxonomy_review", "decision_agent")
COLUMNS = ("vc_id", "episode_id", "stage", "rationale_dimension", "rationale_label", "rationale_activation",
           "direction", "salience", "confidence", "weighted_score", "weighted_call", "agent_call", "agent_likelihood")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def first(*paths):
    return next((p for p in paths if p.is_file()), None)


def case(folder):
    """One pitch's files in either layout, or None when the folder is not a run."""
    if (folder / "run-config.json").is_file() and (folder / "phase1" / "investigation.json").is_file():
        config = read(folder / "run-config.json")["run"]
        return {"vc": config["vc_slug"], "episode": config["episode_slug"],
                "investigation": folder / "phase1" / "investigation.json",
                "decision": first(folder / "phase2" / "decision.json"),
                "coverage": first(folder / "coverage.json", folder / "components" / "coverage.json"),
                "final": first(folder / "lens.json", folder / "components" / "lens.json", folder / "final_lens.json")}
    if (folder / "investigation.json").is_file() and (folder / "decision.json").is_file():
        return {"vc": folder.parent.name, "episode": folder.name,
                "investigation": folder / "investigation.json", "decision": folder / "decision.json",
                "coverage": first(folder / "coverage.json"), "final": first(folder / "final_lens.json")}
    return None


def collect(roots):
    found, warnings = {}, []
    for root in roots:
        for marker in sorted({*root.rglob("investigation.json")}):
            folder = marker.parent.parent if marker.parent.name == "phase1" else marker.parent
            item = case(folder)
            if item is None:
                continue
            if item["decision"] is None:
                warnings.append(f"no decision, skipped: {folder}")
                continue
            key = (item["vc"], item["episode"])
            previous = found.get(key)
            if previous is None or item["decision"].stat().st_mtime > previous["decision"].stat().st_mtime:
                if previous is not None:
                    warnings.append(f"several runs for {key[0]} / {key[1]}; kept {folder}")
                found[key] = item
    return found, warnings


def taxonomy(path):
    value = read(path)
    if isinstance(value, dict):
        return value
    return {row["label"]: row.get("coarse_parent") for row in value}


def score(items):
    return sum(SIGN.get(x.get("direction"), 0.0) * WEIGHT.get(x.get("salience"), 0.5)
               * (1.0 if x.get("confidence") is None else float(x["confidence"])) for x in items)


def top_cutoff(values):
    ranked = sorted(values, reverse=True)
    return ranked[max(1, round(QUOTA * len(ranked))) - 1]


def weighted_calls(scores):
    by_investor = defaultdict(list)
    for key in scores:
        by_investor[key[0]].append(key)
    calls = {}
    for key, value in scores.items():
        own = [scores[k] for k in by_investor[key[0]] if k != key]
        pool = own if len(own) >= MIN_OWN else [scores[k] for k in scores if k != key]
        calls[key] = "In" if pool and value >= top_cutoff(pool) else "Out"
    return calls


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--runs", type=Path, action="append", required=True, help="Folder of runs; repeat for several")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--taxonomy", type=Path, default=Path("proposal-evaluation/data/taxonomy.json"),
                    help="Rationale label -> dimension (taxonomy.json, or a codebook from inputs/taxonomy/)")
    args = ap.parse_args()
    for root in args.runs:
        if not root.is_dir():
            ap.error(f"--runs folder not found: {root}")
    dimension = taxonomy(args.taxonomy)
    cases, warnings = collect(args.runs)
    if not cases:
        ap.error("no runs found")

    stages, decisions = {}, {}
    for key, item in cases.items():
        stages[key] = {
            "investigator": read(item["investigation"]).get("rationales", []),
            "taxonomy_review": read(item["coverage"]).get("additions", []) if item["coverage"] else [],
            "decision_agent": read(item["final"]).get("rationales", []) if item["final"] else [],
        }
        decisions[key] = read(item["decision"])
        for name, field in (("coverage", "taxonomy review"), ("final", "decision agent")):
            if item[name] is None:
                warnings.append(f"no {field} rationales: {key[0]} / {key[1]}")
    scores = {key: score(s["investigator"] + s["taxonomy_review"]) for key, s in stages.items()}
    calls = weighted_calls(scores)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    unknown = set()
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(COLUMNS)
        for key in sorted(stages):
            decision = decisions[key]
            for stage in STAGES:
                for x in stages[key][stage]:
                    label = x.get("taxonomy_label")
                    if label not in dimension:
                        unknown.add(label)
                    writer.writerow([key[0], key[1], stage, dimension.get(label) or "", label, 1,
                                     x.get("direction"), x.get("salience"),
                                     "" if x.get("confidence") is None else x["confidence"],
                                     scores[key], calls[key], decision.get("decision"),
                                     decision.get("investment_likelihood")])
    for line in warnings:
        print(f"warning: {line}", file=sys.stderr)
    if unknown:
        print(f"warning: {len(unknown)} labels missing from {args.taxonomy}: {', '.join(sorted(map(str, unknown)))}",
              file=sys.stderr)
    print(f"{len(stages)} decisions from {len(args.runs)} folder(s) -> {args.out}")


if __name__ == "__main__":
    main()
