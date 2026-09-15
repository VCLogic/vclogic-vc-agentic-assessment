from vc_clone_graph.pitch_cleanup import remove_post_boundary_lines


def test_remove_post_boundary_lines_preserves_pitch_and_removes_reaction():
    text = (
        "Founder: We have growing revenue.\n"
        "Founder: Thank you.\n"
        "Founder: Amazing. You got it.\n"
    )
    turns = [
        {"idx": 10, "speaker": "Founder", "text": "We have growing revenue."},
        {"idx": 11, "speaker": "Founder", "text": "Thank you."},
        {"idx": 12, "speaker": "Investor", "text": "I'm in."},
        {"idx": 13, "speaker": "Founder", "text": "Amazing. You got it."},
    ]

    cleaned, removed, unmapped = remove_post_boundary_lines(text, turns, 12)

    assert cleaned == "Founder: We have growing revenue.\nFounder: Thank you.\n"
    assert removed == [{"turn_index": 13, "line": "Founder: Amazing. You got it."}]
    assert unmapped == []


def test_remove_post_boundary_lines_never_adds_or_drops_unmapped_text():
    text = "Founder: Pitch text normalized elsewhere.\n"

    cleaned, removed, unmapped = remove_post_boundary_lines(text, [], 5)

    assert cleaned == text
    assert removed == []
    assert unmapped == ["Founder: Pitch text normalized elsewhere."]


def test_remove_post_boundary_lines_preserves_repeated_line_seen_before_boundary():
    text = "Founder: Thank you.\n"
    turns = [
        {"idx": 2, "speaker": "Founder", "text": "Thank you."},
        {"idx": 8, "speaker": "Founder", "text": "Thank you."},
    ]

    cleaned, removed, unmapped = remove_post_boundary_lines(text, turns, 5)

    assert cleaned == text
    assert removed == []
    assert unmapped == []
