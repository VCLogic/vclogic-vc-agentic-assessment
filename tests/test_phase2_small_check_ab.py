from scripts.run_phase2_small_check_ab import CALIBRATION_MARKER, revised_prompt


def test_revised_prompt_is_bounded_and_label_blind():
    original = "Original evidence-bearing Phase 2 prompt."

    result = revised_prompt(original)

    assert result.startswith(original + "\n\n")
    assert result.count(CALIBRATION_MARKER) == 1
    added = result.removeprefix(original)
    assert "actual label" not in added.casefold()
    assert "actual_decision" not in added.casefold()
    assert "$25k" in added
    assert "$100k" in added
    assert "even the smallest real check" in added
