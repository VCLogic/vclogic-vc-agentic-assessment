from pathlib import Path

from vc_clone_graph.historical_replay_runner import load_historical_case


ROOT = Path(__file__).resolve().parents[1]
VC = "charles-hudson-precursor-ventures"


def test_load_historical_case_derives_speakers_labels_and_aliases() -> None:
    case = load_historical_case(
        input_root=ROOT / "inputs",
        evaluation_root=ROOT / "evaluation",
        vc_slug=VC,
        vc_name="Charles Hudson",
        episode_slug="18-rowvigor",
    )

    assert case.episode.decision.status == "Out"
    assert case.target_company_aliases == ("Row Vigor", "ROWViGOR")
    assert [row.turn_index for row in case.canary.observed_questions] == [31, 71]
    assert set(case.canary.observed_questions[0].rationale_labels) == {
        "business_viability_concern",
        "product_success_factor_assessment",
    }
    assert case.canary.founder_answers[1].startswith("We have an app-only model")


def test_load_historical_case_never_places_decision_in_canary() -> None:
    case = load_historical_case(
        input_root=ROOT / "inputs",
        evaluation_root=ROOT / "evaluation",
        vc_slug=VC,
        vc_name="Charles Hudson",
        episode_slug="39-this-pitch-is-damn-near-perfect",
    )

    payload = case.canary.model_dump_json()
    assert "I’m going to invest $50,000" not in payload
    assert case.episode.decision.status == "In"
    assert len(case.canary.observed_questions) == 2
    decision_turn = min(row.turn_start for row in case.episode.decision.evidence)
    assert case.panel_founder_statements
    assert all(row.turn_index < decision_turn for row in case.panel_founder_statements)
    assert all(row.speaker not in {"Charles", "Narrator"} for row in case.panel_founder_statements)
    assert "I’m going to invest $50,000" not in " ".join(
        row.text for row in case.panel_founder_statements
    )

