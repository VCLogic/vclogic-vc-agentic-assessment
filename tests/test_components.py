from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import pytest

from vc_clone_graph.components.config import ConsolidateSettings, CoverageSettings
from vc_clone_graph.components.consolidate import (
    TALLY_VERSION,
    build_calibration,
    consolidate,
    decide,
    evaluate_history,
    tally,
    top_cutoff,
)
from vc_clone_graph.components.coverage import build_prompt, run_coverage
from vc_clone_graph.components.run_inputs import load_run
from vc_clone_graph.providers.base import GenerationResult, Usage

PITCH = "We sell to schools. We are raising $2 million at a $20 million valuation.\nRevenue grew 3x."
TAXONOMY = [
    {"label": "market_size_assessment", "definition": "Market size.", "coarse_parent": "market_opportunity"},
    {"label": "valuation_concern", "definition": "Price too high.", "coarse_parent": "deal_terms_valuation"},
    {"label": "timing_assessment", "definition": "Why now.", "coarse_parent": "timing"},
]


def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def make_run(root: Path, *, vc="vc-a", episode="ep-1", decision="In", rationales=None, contract="v4.1") -> Path:
    pitch = root / "inputs" / f"{vc}-{episode}.txt"
    pitch.parent.mkdir(parents=True, exist_ok=True)
    pitch.write_text(PITCH, encoding="utf-8")
    _write(root / "inputs" / "taxonomy" / "codebook.json", TAXONOMY)
    run = root / "runs" / vc / episode
    _write(run / "run-config.json", {"run": {
        "vc_slug": vc, "episode_slug": episode, "contract_version": contract,
        "input_root": str(root / "inputs"), "taxonomy_path": "taxonomy/codebook.json"}})
    _write(run / "input-provenance.json", {
        "pitch_path": str(pitch), "pitch_sha256": sha256(pitch.read_bytes()).hexdigest()})
    _write(run / "assessment-policy.json", {
        "target_context": {"investor_name": "Investor A"},
        "evidence": [{"evidence_id": "W-1", "heading": "Checks", "text": "Writes $25k checks at low valuations."}]})
    _write(run / "phase1" / "investigation.json", {"rationales": rationales if rationales is not None else [
        {"rationale_id": "R1", "taxonomy_label": "market_size_assessment", "direction": "positive",
         "salience": "primary", "confidence": 0.8, "justification": "Big market."}]})
    _write(run / "phase2" / "decision.json", {
        "decision": decision, "investment_likelihood": 0.6, "controlling_rationale_ids": ["R1"]})
    return run


class FakeProvider:
    def __init__(self, parsed):
        self.parsed = parsed
        self.requests = []

    def generate(self, request):
        self.requests.append(request)
        return GenerationResult(parsed=self.parsed, content=json.dumps(self.parsed),
                                usage=Usage(input_tokens=10), elapsed_seconds=0.0, raw_metadata={})


def calibration(cutoff: float, n: int = 10, includes_coverage: bool = False) -> dict:
    tallies = [cutoff] * n
    return {"tally_version": TALLY_VERSION, "min_vc_cases": 8, "quota": 0.3, "includes_coverage": includes_coverage,
            "global": {"n": n, "cutoff": cutoff, "tallies": tallies},
            "investors": {"vc-a": {"n": n, "cutoff": cutoff, "tallies": tallies}}}


def test_prompt_sends_only_uncovered_labels_and_compact_phase1(tmp_path):
    run = load_run(make_run(tmp_path))
    prompt, labels = build_prompt(run, CoverageSettings())
    assert labels == ["valuation_concern", "timing_assessment"]
    assert "Big market." not in prompt  # Phase 1 justifications are not sent
    assert "- market_size_assessment (positive, primary)" in prompt
    assert "W-1" in prompt and PITCH in prompt


def test_coverage_keeps_grounded_uncovered_additions_and_rejects_the_rest(tmp_path):
    run = load_run(make_run(tmp_path))
    provider = FakeProvider({"review_note": "", "additions": [
        {"taxonomy_label": "valuation_concern", "direction": "negative", "salience": "primary",
         "confidence": 0.7, "pitch_quote": "raising $2  million at a $20 million valuation",
         "policy_evidence_ids": ["W-1", "W-9"], "justification": "Above typical caps."},
        {"taxonomy_label": "timing_assessment", "direction": "positive", "salience": "secondary",
         "confidence": 0.5, "pitch_quote": "the market is exploding right now", "policy_evidence_ids": [],
         "justification": "Invented quote."},
        {"taxonomy_label": "market_size_assessment", "direction": "positive", "salience": "primary",
         "confidence": 0.9, "pitch_quote": "We sell to schools", "policy_evidence_ids": [], "justification": ""},
    ]})
    result = run_coverage(run, CoverageSettings(), provider)
    assert [a["taxonomy_label"] for a in result["additions"]] == ["valuation_concern"]
    assert result["additions"][0]["policy_evidence_ids"] == ["W-1"]
    assert result["additions"][0]["unknown_policy_evidence_ids"] == ["W-9"]
    assert {r["rejection_reason"] for r in result["rejected"]} == {
        "quote_not_found_in_pitch", "label_already_covered_or_unknown"}
    assert provider.requests[0].schema["properties"]["additions"]["items"]["properties"][
        "taxonomy_label"]["enum"] == ["valuation_concern", "timing_assessment"]


def test_coverage_skips_call_when_everything_is_covered(tmp_path):
    rationales = [{"rationale_id": f"R{i}", "taxonomy_label": t["label"], "direction": "positive",
                   "salience": "secondary", "confidence": 0.5} for i, t in enumerate(TAXONOMY)]
    provider = FakeProvider({})
    result = run_coverage(load_run(make_run(tmp_path, rationales=rationales)), CoverageSettings(), provider)
    assert result["status"] == "skipped_no_uncovered_labels" and provider.requests == []


def test_v44_runs_are_refused(tmp_path):
    with pytest.raises(ValueError, match="v4.4"):
        load_run(make_run(tmp_path, contract="v4.4"))


def test_tally_and_cutoff():
    items = [{"direction": "positive", "salience": "primary", "confidence": 0.8},
             {"direction": "negative", "salience": "secondary", "confidence": 1.0},
             {"direction": "neutral", "salience": "primary", "confidence": 1.0}]
    assert tally(items) == pytest.approx(0.3)
    assert top_cutoff([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 0.3) == 8


def test_decision_rules():
    assert decide("veto", True, False) is False
    assert decide("veto", True, True) is True
    assert decide("tally", False, True) is True
    assert decide("phase2", True, False) is True


def test_consolidate_vetoes_unsupported_phase2_in_and_averages_runs(tmp_path):
    weak = [{"rationale_id": "R1", "taxonomy_label": "market_size_assessment", "direction": "positive",
             "salience": "secondary", "confidence": 0.2}]
    runs = [load_run(make_run(tmp_path / str(i), rationales=weak)) for i in range(2)]
    result = consolidate(runs, calibration=calibration(1.0), rule="veto")
    assert result["decision"] == "Out"
    assert result["phase2"]["decision"] == "In"
    assert "phase2_tally_disagreement" in result["flags"]
    assert result["scores"]["phase1_tally"] == pytest.approx(0.1)


def test_consolidate_reports_coverage_but_scores_it_only_when_enabled(tmp_path):
    run = load_run(make_run(tmp_path))
    coverage = {"vc_slug": "vc-a", "episode_slug": "ep-1", "status": "completed", "additions": [
        {"addition_id": "C1", "taxonomy_label": "valuation_concern", "direction": "negative",
         "salience": "primary", "confidence": 1.0, "pitch_quote": "q"}]}
    off = consolidate([run], calibration=calibration(0.5), coverage=coverage, rule="tally")
    on = consolidate([run], calibration=calibration(0.5, includes_coverage=True), coverage=coverage, rule="tally",
                     use_coverage=True)
    assert off["decision"] == "In" and "coverage_reported_not_scored" in off["flags"]
    assert on["decision"] == "Out" and on["scores"]["tally"] == pytest.approx(-0.2)
    assert on["rationales"]["coverage_additions"][0]["taxonomy_label"] == "valuation_concern"


def test_consolidate_refuses_calibration_built_without_coverage(tmp_path):
    run = load_run(make_run(tmp_path))
    with pytest.raises(ValueError, match="use_coverage"):
        consolidate([run], calibration=calibration(0.5), use_coverage=True)


def test_consolidate_flags_missing_coverage_when_scored(tmp_path):
    run = load_run(make_run(tmp_path))
    result = consolidate([run], calibration=calibration(0.5, includes_coverage=True), use_coverage=True)
    assert "coverage_missing_tally_biased_low" in result["flags"]


def test_calibration_with_coverage_shifts_cutoffs_and_requires_every_case():
    cases = {("vc-a", f"e{i}"): [{"tally": float(i), "actual": "In"}] for i in range(10)}
    cal = build_calibration(cases, coverage_tallies={k: 1.0 for k in cases})
    assert cal["includes_coverage"] is True and cal["investors"]["vc-a"]["cutoff"] == 8.0
    with pytest.raises(ValueError, match="no coverage"):
        build_calibration(cases, coverage_tallies={("vc-a", "e0"): 1.0})


def test_consolidate_rejects_mismatched_coverage(tmp_path):
    run = load_run(make_run(tmp_path))
    with pytest.raises(ValueError, match="different"):
        consolidate([run], calibration=calibration(0.5), coverage={"vc_slug": "vc-b", "episode_slug": "ep-1"})


def test_calibration_uses_no_labels_and_falls_back_to_global():
    cases = {("vc-a", f"e{i}"): [{"tally": float(i), "actual": "In"}] for i in range(10)}
    cases[("vc-small", "x")] = [{"tally": 100.0, "actual": "Out"}]
    cal = build_calibration(cases, quota=0.3, min_vc_cases=8)
    assert cal["uses_outcome_labels"] is False
    assert cal["investors"]["vc-a"]["cutoff"] == 7.0
    from vc_clone_graph.components.consolidate import cutoff_for
    assert cutoff_for(cal, "vc-small")[1] == "global"


def test_leave_one_out_evaluation_runs():
    cases = {("vc-a", f"e{i}"): [{"tally": float(i), "likelihood": 0.5, "verdict": "In" if i % 2 else "Out",
                                   "actual": "In" if i >= 7 else "Out"}] for i in range(10)}
    report = evaluate_history(cases, quota=0.3, min_vc_cases=5)
    assert report["ranking_auc"]["tally_mean_of_runs"] == 1.0
    assert report["decisions"]["tally_mean_of_runs"]["in_recall"] == 1.0


def test_settings_defaults_match_validated_setup():
    consolidate_settings = ConsolidateSettings()
    assert CoverageSettings().enabled is True and consolidate_settings.enabled is True
    assert consolidate_settings.rule == "tally" and consolidate_settings.use_coverage is True


def _hook_config(run: Path):
    from types import SimpleNamespace
    return SimpleNamespace(
        run=SimpleNamespace(output_root=str(run.parent), episode_slug=run.name),
        resolve_path=lambda value: Path(value),
    )


def test_pipeline_hook_writes_consolidated_output(tmp_path, monkeypatch):
    from vc_clone_graph.cli import _run_components

    run = make_run(tmp_path)
    cal = tmp_path / "cal.json"
    cal.write_text(json.dumps(calibration(0.5)), encoding="utf-8")
    components = tmp_path / "components.toml"
    components.write_text(f'[coverage]\nenabled = false\n[lens]\nenabled = false\n[consolidate]\n'
                          f'calibration_path = "{cal.name}"\nrule = "veto"\nuse_coverage = false\n')
    monkeypatch.chdir(tmp_path)
    _run_components(_hook_config(run), components)
    result = json.loads((run / "components" / "consolidated.json").read_text())
    assert result["decision"] == "In"
    assert not (run / "components" / "coverage.json").exists()


def test_pipeline_hook_runs_lens_after_the_decision(tmp_path, monkeypatch):
    from vc_clone_graph.cli import _run_components
    from vc_clone_graph.components import cli as components_cli
    from vc_clone_graph.components import lens as lens_module

    run = make_run(tmp_path)
    cal = tmp_path / "cal.json"
    cal.write_text(json.dumps(calibration(0.5)), encoding="utf-8")
    components = tmp_path / "components.toml"
    components.write_text(f'[coverage]\nenabled = false\n[consolidate]\ncalibration_path = "{cal.name}"\n'
                          'use_coverage = false\n[lens]\ninclude_memory = false\n')
    provider = FakeProvider({"investor_focus": "Market.", "additions": [], "reviews": [
        {"candidate_id": "R1", "action": "keep", "direction": "negative", "salience": "primary"}]})
    monkeypatch.setattr(components_cli, "openrouter_provider", lambda settings: provider)
    monkeypatch.setattr(lens_module, "load_precedents", lambda run, settings, workspace: [])
    monkeypatch.chdir(tmp_path)
    _run_components(_hook_config(run), components)
    lens = json.loads((run / "components" / "lens.json").read_text())
    assert lens["variant"] == "contrastive" and lens["panel"] == []
    assert [(r["rationale_id"], r["direction"]) for r in lens["rationales"]] == [("R1", "negative")]
    assert (run / "components" / "consolidated.json").is_file()


def test_pipeline_hook_failure_is_recorded_not_raised(tmp_path, capsys):
    from vc_clone_graph.cli import _run_components

    run = make_run(tmp_path)
    components = tmp_path / "components.toml"
    components.write_text('[coverage]\nenabled = false\n[consolidate]\ncalibration_path = "missing.json"\n')
    _run_components(_hook_config(run), components)
    assert "error" in json.loads((run / "components" / "error.json").read_text())
    assert "components failed" in capsys.readouterr().err


def _cli_config(kind: str, contract: str = "v4.1"):
    from types import SimpleNamespace
    return SimpleNamespace(provider=SimpleNamespace(kind=kind), run=SimpleNamespace(contract_version=contract))


def test_components_run_by_default_only_for_real_non_v44_runs():
    from vc_clone_graph.cli import components_apply_by_default

    assert components_apply_by_default(_cli_config("openrouter"))
    assert not components_apply_by_default(_cli_config("fake"))
    assert not components_apply_by_default(_cli_config("openrouter", "v4.4"))


@pytest.mark.parametrize("flags, expected", [([], "default"), (["--no-components"], None)])
def test_main_runs_default_components_unless_disabled(monkeypatch, flags, expected):
    from vc_clone_graph import cli

    calls = []
    monkeypatch.setattr(cli, "load_config", lambda path: _cli_config("openrouter"))
    monkeypatch.setattr(cli, "command_run", lambda config, resume=False: None)
    monkeypatch.setattr(cli, "run_default_components", lambda config: calls.append("default"))
    assert cli.main(["run", "--config", "x.toml", *flags]) == 0
    assert calls == ([expected] if expected else [])


def _coverage(tally_direction: str, label: str = "valuation_concern", status: str = "completed") -> dict:
    return {"vc_slug": "vc-a", "episode_slug": "ep-1", "status": status, "additions": [
        {"addition_id": "C1", "taxonomy_label": label, "direction": tally_direction,
         "salience": "primary", "confidence": 1.0, "pitch_quote": "q"}]}


def test_consolidate_averages_coverage_over_runs_and_flags_gaps(tmp_path):
    runs = [load_run(make_run(tmp_path / str(i))) for i in range(2)]
    cal = calibration(0.5, includes_coverage=True)
    both = consolidate(runs, calibration=cal, coverage=[_coverage("negative"), _coverage("positive")], use_coverage=True)
    assert both["scores"]["coverage_tally"] == pytest.approx(0.0)
    assert len(both["rationales"]["coverage_additions"]) == 2
    one = consolidate(runs, calibration=cal, coverage=[_coverage("negative")], use_coverage=True)
    assert "coverage_missing_for_some_runs" in one["flags"]


def test_same_model_ignores_openrouter_prefix():
    from vc_clone_graph.components.consolidate import same_model

    assert same_model("gpt-5.6-luna", "openai/gpt-5.6-luna")
    assert not same_model("openai/gpt-5.6-sol", "openai/gpt-5.6-luna")


def test_coverage_tallies_require_every_run(tmp_path):
    from vc_clone_graph.components.cli import _backfill_path, _coverage_tallies

    paths = []
    for i in range(2):
        p1 = make_run(tmp_path / str(i)) / "phase1" / "investigation.json"
        paths.append(str(p1))
    evaluation = {"results": [{"execution_status": "completed", "input_mode": "founders-only", "model": "openai/gpt-5.6-luna",
                               "vc_slug": "vc-a", "episode_slug": "ep-1", "phase1_path": p} for p in paths]}
    root = tmp_path / "backfill"
    _write(_backfill_path(root, "vc-a", "ep-1", paths[0]), _coverage("negative"))
    assert _coverage_tallies(evaluation, root, "openai/gpt-5.6-luna") == {}
    _write(_backfill_path(root, "vc-a", "ep-1", paths[1]), _coverage("positive"))
    assert _coverage_tallies(evaluation, root, "openai/gpt-5.6-luna") == {("vc-a", "ep-1"): [-1.0, 1.0]}


def test_investor_quota_moves_the_cutoff_not_the_tally(tmp_path):
    run = load_run(make_run(tmp_path))
    spread = calibration(0.0)
    spread["investors"]["vc-a"]["tallies"] = [i / 10 for i in range(11)]  # 0.0 ... 1.0
    default = consolidate([run], calibration=spread, rule="tally")
    stricter = consolidate([run], calibration=spread, rule="tally", quota=0.1)
    looser = consolidate([run], calibration=spread, rule="tally", quota=0.5)
    assert stricter["scores"]["tally"] == default["scores"]["tally"] == looser["scores"]["tally"]
    assert stricter["decision"] == "Out" and stricter["scores"]["cutoff"] == pytest.approx(1.0)
    assert looser["decision"] == "In" and looser["scores"]["quota_source"] == "investor_quota"
    assert default["scores"]["quota_source"] == "calibration"


def test_investor_quota_setting_is_validated():
    from vc_clone_graph.components.config import ConsolidateSettings
    assert ConsolidateSettings(investor_quota={"vc-a": 0.2}).investor_quota == {"vc-a": 0.2}
    with pytest.raises(ValueError):
        ConsolidateSettings(investor_quota={"vc-a": 1.5})
