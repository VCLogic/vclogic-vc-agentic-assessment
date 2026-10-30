"""Freeze the inputs of the proposal evaluation into proposal-evaluation/data/.

Reads the local evaluation records (evaluation/, outputs/, rationale_annotation/) once and writes
small coded files: rationale labels, directions, salience, confidence, calls and likelihoods. No
transcripts, pitches or quotes are copied. Run from the repository root:

    uv run python proposal-evaluation/build_data.py

evaluate.py reads only data/, so the results can be reproduced without the raw records.
"""
import csv
import glob
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, "src")
from vc_clone_graph.components.cli import _backfill_path, _latest_runs, _lens_path, _read_evaluation

OUT = Path("proposal-evaluation/data")
MODEL = "openai/gpt-5.6-luna"
KEEP = ("taxonomy_label", "direction", "salience", "confidence")


def items(rationales, stage):
    return [{**{k: r.get(k) for k in KEEP}, "stage": stage} for r in rationales]


def mentions(ms):
    return [{"label": m["rationale_label"], "dimension": m["rationale_category"], "direction": m["direction"],
             "decision_link": m["decision_link"]} for m in ms]


def write_jsonl(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True) + "\n")


def taxonomy():
    cat = {}
    for name in ("codebook_v2.json", "codebook_v_final.json"):
        for e in json.load(open(f"inputs/taxonomy/{name}")):
            cat.setdefault(e["label"], e.get("coarse_parent"))
    return cat


def decisions():
    ev = _read_evaluation(Path("evaluation/pitchshow/results/automated-evaluation/latest.json"))
    actual = {(r["vc_slug"], r["episode_slug"]): r["actual_pitch_decision"] for r in ev["results"]
              if r.get("actual_pitch_decision") in ("In", "Out")}
    rows = []
    for (vc, ep), run in sorted(_latest_runs(ev, MODEL).items()):
        p1 = json.load(open(run["phase1_path"]))["rationales"]
        cp = _backfill_path(Path("outputs/components-backfill"), vc, ep, run["phase1_path"])
        cov = json.load(open(cp))["additions"] if cp.is_file() else []
        lens = json.load(open(_lens_path(Path("outputs/components-lens"), "contrastive", vc, ep)))["rationales"]
        a = Path(f"rationale_annotation/annotations/{vc}/{ep}.json")
        ms = None
        if a.is_file():
            ms = [m for m in json.load(open(a))["rationale_mentions"]
                  if m["scope"] in ("original_pitch", "post_founder_debrief") and m["direction"] != "unclear"]
        rows.append({
            "investor": vc, "episode": ep, "actual": actual[(vc, ep)],
            "investigator_rationales": items(p1, "investigator") + items(cov, "taxonomy_review"),
            "final_rationales": items(lens, "decision_agent"),
            "agent_call": run["verdict"], "agent_likelihood": float(run["likelihood"]),
            "reference_mentions": None if ms is None else mentions(ms),
        })
    return rows


def second_coding(rows):
    """The earlier, independent coding of the same on-air statements (for the agreement figures)."""
    out = []
    for r in rows:
        if r["reference_mentions"] is None:
            continue
        for root in ("live-founder-v2", "records"):
            q = Path("evaluation/pitchshow/rationale-corpus") / root / r["investor"] / f"{r['episode']}.json"
            if q.exists():
                o = json.load(open(q))["rationales"]
                out.append({"investor": r["investor"], "episode": r["episode"],
                            "mentions": [{"label": x["rationale_label"], "direction": x["direction"]} for x in o]})
                break
    return out


def single_call(rows):
    keys = {(r["investor"], r["episode"]) for r in rows}
    prog = json.load(open("evaluation/pitchshow/results/prompt-only-luna-all-2026-10-21-progress.json"))
    out = []
    for c in prog["cases"]:
        k = (c["vc_slug"], c["episode_slug"])
        if c["status"] != "completed" or k not in keys:
            continue
        resp = json.load(open(f"outputs/prompt-only-luna-all-2026-10-21/cases/{c['case_id']}/provider-response.json"))
        d = (resp.get("parsed") or {}).get("decision") or {}
        if "decision" in d:
            out.append({"investor": k[0], "episode": k[1], "call": d["decision"],
                        "likelihood": float(d["investment_likelihood"])})
    status = {}
    for c in prog["cases"]:
        status[c["status"]] = status.get(c["status"], 0) + 1
    return out, {"planned": len(prog["cases"]), "status": status,
                 "cost_usd": round(float(prog.get("recorded_cost_usd") or 0), 2)}


def sol_pilots():
    pilots = []
    for n, name in enumerate(("gpt6-sol-paired16-2026-10-20", "gpt6-sol-round2-16-2026-10-20"), 1):
        exp = json.load(open(f"evaluation/pitchshow/results/{name}.json"))
        prog = json.load(open(f"evaluation/pitchshow/results/{name}-progress.json"))
        sol = {c["case_id"]: c for c in prog["cases"]}
        cases = [{"investor": c["vc_slug"], "episode": c["episode_slug"], "stratum": c["stratum"],
                  "actual": c["actual_decision"], "luna_call": c["baseline_prediction"],
                  "sol_call": sol[c["case_id"]]["predicted_decision"]} for c in exp["cases"]]
        pilots.append({"pilot": n, "sol_cost_usd": sum(float(c.get("cost_usd") or 0) for c in prog["cases"]),
                       "cases": cases})
    return pilots


def funnel(assessed):
    summary = json.load(open("evaluation/pitchshow/summary.json"))["investors"]
    out = []
    for vc in sorted(summary):
        s = summary[vc]
        missing = set(s.get("missing_transcripts") or [])
        for a in json.load(open(f"evaluation/pitchshow/investors/{vc}/annotations.json")):
            out.append({"investor": vc, "episode": a["episode_slug"], "has_transcript": a["episode_slug"] not in missing,
                        "on_panel": a.get("episode_kind") == "pitch", "decision": a.get("pitch_window_decision"),
                        "passed_leakage_check": bool(a.get("evaluation_eligible")),
                        "assessed": (vc, a["episode_slug"]) in assessed})
    return out, {vc: {"episodes": s["episodes"], "available_transcripts": s["available_transcripts"]}
                 for vc, s in summary.items()}


def memory_sizes(investors):
    """Evidence excerpts, context entries and indexed sources per Investment Memory."""
    out = {}
    for vc in sorted(investors):
        w = Path("inputs/wiki") / vc
        ev = {line.split("]")[0] for p in (w / "evidence").glob("*.md") for line in open(p) if line.startswith("[ev:")}
        ctx = w / "context.md"
        src = w / "sources.md"
        out[vc] = {"evidence_excerpts": len(ev),
                   "context_entries": len(set(re.findall(r"^#+\s.*?\b(ctx[:\-][\w\-]+)", ctx.read_text(), re.M)))
                   if ctx.exists() else None,
                   "sources_md_present": src.exists()}
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(taxonomy(), open(OUT / "taxonomy.json", "w"), indent=1, sort_keys=True)
    rows = decisions()
    write_jsonl(OUT / "decisions.jsonl", rows)
    json.dump(memory_sizes({r["investor"] for r in rows}), open(OUT / "memory_sizes.json", "w"), indent=1)
    write_jsonl(OUT / "reference_second_coding.jsonl", second_coding(rows))
    sc, sc_status = single_call(rows)
    write_jsonl(OUT / "single_call.jsonl", sc)
    json.dump(sc_status, open(OUT / "single_call_run.json", "w"), indent=1)
    json.dump(sol_pilots(), open(OUT / "sol_pilots.json", "w"), indent=1)
    fn, counts = funnel({(r["investor"], r["episode"]) for r in rows})
    write_jsonl(OUT / "funnel.jsonl", fn)
    json.dump(counts, open(OUT / "funnel_counts.json", "w"), indent=1, sort_keys=True)
    with open("outputs/expressed-interest/on-panel-unobserved-responses.csv") as f, \
            open(OUT / "no_decision_responses.csv", "w", newline="") as g:
        w = csv.writer(g)
        w.writerow(["episode", "investor", "category"])
        for r in csv.DictReader(f):
            w.writerow([r["episode"], r["investor"], r["category"]])
    print(f"{len(rows)} decisions, {len(sc)} single-call decisions -> {OUT}")


if __name__ == "__main__":
    main()
