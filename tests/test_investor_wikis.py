from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_elizabeth_wiki_defines_source_grounded_founder_conviction() -> None:
    persona = (
        ROOT / "inputs/wiki/elizabeth-yin-hustle-fund/persona.md"
    ).read_text(encoding="utf-8")

    assert "## founder_conviction_calibration" in persona
    section = persona.split("## founder_conviction_calibration", 1)[1].split(
        "\n## ", 1
    )[0]
    assert "[ev:yin-0001]" in section
    assert "[ev:yin-0002]" in section
    assert "[ev:yin-0007]" in section
    assert "before meaningful revenue" in section
    assert "does not rescue" in section
    assert "structurally weak" in section
