import pytest
from pydantic import ValidationError

from prompt import BaselineContext, PrecedentContext, WikiDocument, build_prompt
from schema import OneShotBaselineResponse, baseline_json_schema


def payload():
    return {
        "schema_version": "one-shot-baseline-v1",
        "episode_slug": "39-example",
        "rationales": [{
            "rationale_id": "R1", "label": "founder_execution",
            "direction": "positive", "salience": "primary", "confidence": .8,
            "pitch_evidence": ["The founder has built and sold before."],
            "wiki_evidence": ["Charles repeatedly values demonstrated execution."],
            "precedent_evidence": ["18-example: Charles backed execution evidence."],
            "interpretation": "Execution is demonstrated rather than asserted."
        }],
        "decision": {
            "decision": "In", "investment_likelihood": .7,
            "decision_confidence": .75, "ranking_score": .8,
            "check_tier": "$50k-$100k", "decision_endpoint": "any_check",
            "recommended_check_tier": "$50k-$100k",
            "controlling_rationale_ids": ["R1"],
            "any_check": {"decision": "In", "likelihood": .7, "confidence": .75,
                "supporting_rationale_ids": ["R1"], "opposing_rationale_ids": [],
                "fatal_constraint_present": False, "optionality_explanation": "A small check preserves upside.",
                "strongest_counterargument": {"argument": "Evidence is incomplete.", "rationale_ids": ["R1"], "response": "Execution offsets it."},
                "reversal_conditions": ["Material conflict"]},
            "standard_check": {"decision": "Out", "likelihood": .35, "confidence": .6,
                "supporting_rationale_ids": ["R1"], "opposing_rationale_ids": [], "failure_rationale_ids": ["R1"],
                "market_gate": {"status": "unresolved", "controlling_rationale_ids": ["R1"], "explanation": "Market proof is incomplete."},
                "strongest_counterargument": {"argument": "Execution may justify a check.", "rationale_ids": ["R1"], "response": "Not at standard size."},
                "upgrade_conditions": ["Market validation"]},
            "risk_ledger": [{"rationale_id": "R1", "risk_type": "missing_evidence", "controlling_for_any_check": False, "counterevidence": [], "explanation": "Market proof missing."}],
            "rationale_assessments": [{"rationale_id": "R1", "effective_direction": "positive", "decision_weight": "decisive", "assessment": "Strong execution.", "counterevidence": []}],
            "deliberation_steps": [
                {"step_id": f"D{i}", "endpoint": "any_check" if i < 3 else "standard_check", "stage": stage,
                 "question": "What controls the decision?", "rationale_ids": ["R1"], "evidence_assessment": "Execution is supported.",
                 "likelihood_before": .5, "likelihood_after": .7, "effect": "raises", "check_tier_implication": "small check", "decision_update": "Move toward In."}
                for i, stage in enumerate(("assessment", "opposing_case", "decision"), 1)
            ],
            "strongest_counterargument": "Market evidence is incomplete.", "unresolved_questions": ["Market depth"],
            "reversal_conditions": ["Conflict"], "feedback": "Promising but validate market.",
            "stable": True, "decisive_precedents": ["18-example"], "exception_analogies": []
        }
    }


def test_contract_rejects_unknown_rationale_references():
    bad = payload()
    bad["decision"]["controlling_rationale_ids"] = ["R2"]
    with pytest.raises(ValidationError, match="unknown baseline rationale"):
        OneShotBaselineResponse.model_validate(bad)


def test_contract_validates_taxonomy_and_binds_episode_in_schema():
    value = OneShotBaselineResponse.model_validate(payload())
    value.validate_taxonomy({"founder_execution"})
    with pytest.raises(ValueError, match="unknown taxonomy label"):
        value.validate_taxonomy({"market_size"})
    schema = baseline_json_schema("39-example", {"founder_execution"})
    assert schema["properties"]["episode_slug"]["const"] == "39-example"


def test_prompt_contains_all_sources_and_explicit_leakage_boundary():
    context = BaselineContext(
        episode_slug="39-example", investor_name="Charles Hudson", current_pitch="PITCH ONLY",
        wiki=(WikiDocument(path="persona.md", content="FULL PERSONA"),),
        taxonomy=({"label": "founder_execution", "definition": "execution"},),
        precedents=tuple(PrecedentContext(f"{i}-prior", f"TRANSCRIPT {i}", "In" if i % 2 else "Out", f"DECISION {i}", .9-i/100) for i in range(1, 6)),
    )
    rendered = build_prompt(context)
    assert "PITCH ONLY" in rendered and "FULL PERSONA" in rendered
    assert rendered.count("BEGIN PRECEDENT EPISODE") == 5
    assert "Do not infer or search for the target episode's actual decision" in rendered
    assert "private chain-of-thought" in rendered
