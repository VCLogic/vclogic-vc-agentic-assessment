import csv
import json
import os
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "proposal-evaluation" / "build_predictions.py"


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _rationale(label, direction, confidence=0.8):
    return {"taxonomy_label": label, "direction": direction, "salience": "primary", "confidence": confidence}


def test_case_folders_and_live_runs_become_rationale_rows(tmp_path):
    runs = tmp_path / "runs"
    case = runs / "ada" / "ep-1"
    _write(case / "investigation.json", {"rationales": [_rationale("founder_market_fit", "positive")]})
    _write(case / "coverage.json", {"additions": [_rationale("market_size_assessment", "negative", 0.5)]})
    _write(case / "final_lens.json", {"rationales": [_rationale("founder_market_fit", "positive")]})
    _write(case / "decision.json", {"decision": "In", "investment_likelihood": 0.7})

    stale = runs / "old" / "ep-2"
    _write(stale / "run-config.json", {"run": {"vc_slug": "ada", "episode_slug": "ep-2"}})
    _write(stale / "phase1" / "investigation.json", {"rationales": [_rationale("founder_market_fit", "negative")]})
    _write(stale / "phase2" / "decision.json", {"decision": "Out", "investment_likelihood": 0.1})
    live = runs / "new" / "ep-2"
    _write(live / "run-config.json", {"run": {"vc_slug": "ada", "episode_slug": "ep-2"}})
    _write(live / "phase1" / "investigation.json", {"rationales": [_rationale("founder_market_fit", "negative")]})
    _write(live / "phase2" / "decision.json", {"decision": "Out", "investment_likelihood": 0.2})
    _write(live / "components" / "lens.json", {"rationales": [_rationale("market_size_assessment", "negative")]})
    os.utime(stale / "phase2" / "decision.json", (1, 1))  # the newer run must win

    taxonomy = tmp_path / "taxonomy.json"
    _write(taxonomy, {"founder_market_fit": "founder_team", "market_size_assessment": "market_opportunity"})
    out = tmp_path / "predictions.csv"
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--runs", str(runs), "--out", str(out), "--taxonomy", str(taxonomy)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "no taxonomy review rationales: ada / ep-2" in result.stderr

    rows = list(csv.DictReader(out.open(encoding="utf-8")))
    assert [(r["episode_id"], r["stage"], r["rationale_dimension"]) for r in rows] == [
        ("ep-1", "investigator", "founder_team"),
        ("ep-1", "taxonomy_review", "market_opportunity"),
        ("ep-1", "decision_agent", "founder_team"),
        ("ep-2", "investigator", "founder_team"),
        ("ep-2", "decision_agent", "market_opportunity"),
    ]
    first, last = rows[0], rows[-1]
    # 0.8 x primary for the positive, minus 0.5 x primary for the negative addition.
    assert float(first["weighted_score"]) == 0.8 - 0.5
    assert (first["weighted_call"], first["agent_call"], first["agent_likelihood"]) == ("In", "In", "0.7")
    assert (float(last["weighted_score"]), last["weighted_call"], last["agent_likelihood"]) == (-0.8, "Out", "0.2")
