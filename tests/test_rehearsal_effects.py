import pytest

from vc_clone_graph.rehearsal_effects import enforce_evidence_effect_delta


@pytest.mark.parametrize("effect", ["clarification", "unresolved"])
def test_non_material_effects_preserve_score(effect: str) -> None:
    result = enforce_evidence_effect_delta(
        effect=effect,
        before=0.24,
        proposed_after=0.16,
        supporting_excerpt=None,
    )

    assert result.after == 0.24
    assert result.findings == ("non_material_effect_preserved_assessment",)


def test_material_effect_requires_supporting_answer_excerpt() -> None:
    result = enforce_evidence_effect_delta(
        effect="new_negative",
        before=0.24,
        proposed_after=0.16,
        supporting_excerpt=None,
    )

    assert result.after == 0.24
    assert result.findings == ("material_effect_missing_answer_excerpt",)


def test_supported_material_effect_can_move_score() -> None:
    result = enforce_evidence_effect_delta(
        effect="new_positive",
        before=0.24,
        proposed_after=0.38,
        supporting_excerpt="Three customers expanded.",
    )

    assert result.after == 0.38
    assert result.findings == ()


def test_probabilities_are_clamped() -> None:
    result = enforce_evidence_effect_delta(
        effect="contradiction",
        before=0.24,
        proposed_after=-0.4,
        supporting_excerpt="The revenue figure was incorrect.",
    )

    assert result.after == 0.0
    assert result.findings == ("proposed_likelihood_clamped",)
