from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import pytest

from vc_clone_graph.precedent_builder import (
    build_precedent_corpus,
    main,
    parse_speaker_turns,
)
from vc_clone_graph.firewall import validate_precedent_corpus
from vc_clone_graph.precedents import PrecedentCorpus


ROOT = Path(__file__).resolve().parents[1]


def _write_source(root: Path, slug: str, transcript: str | None) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{slug}.json"
    path.write_text(
        json.dumps({"transcript": transcript}, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def _write_ledger(path: Path, rows: list[dict[str, object]]) -> Path:
    path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    return path


def _audit(slug: str, status: str = "In", **overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "episode_slug": slug,
        "pitch_window_decision": status,
        "decision_context": "initial_panel",
        "evidence_quote": "I think I'm in.",
        "audit_notes": "Exact audited response.",
    }
    row.update(overrides)
    return row


def test_builder_binds_conditional_in_to_exact_source_and_preserves_condition(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    slug = "41-example"
    _write_source(
        source,
        slug,
        "Founder: We ship packages.\nCharles: I think I’m in.\nBut I need to make sure I'm de-conflicted.",
    )
    ledger = _write_ledger(
        tmp_path / "labels.json",
        [
            _audit(
                slug,
                evidence_quote="I think I'm in. But I need to make sure I’m de-conflicted.",
                condition="portfolio de-confliction",
                check_tier="$50k-$100k",
            )
        ],
    )

    build_precedent_corpus(source, ledger, tmp_path / "out", ("Charles",))

    row = json.loads((tmp_path / "out" / "records" / f"{slug}.json").read_text())
    assert row["decision"]["status"] == "In"
    assert row["decision"]["conditions"] == ["portfolio de-confliction"]
    assert row["decision"]["check_tier"] == "$50k-$100k"
    assert row["decision"]["evidence"] == [
        {
            "turn_end": 1,
            "turn_start": 1,
            "text": "I think I’m in.\nBut I need to make sure I'm de-conflicted.",
        }
    ]
    assert row["investor_present"] is True


def test_builder_prefers_validated_target_turn_when_speakers_repeat_quote(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    slug = "4-example"
    _write_source(
        source,
        slug,
        "Founder: We build headsets.\n"
        "Howie: I'm going to pass.\n"
        "Founder: Understood.\n"
        "Jillian: I'm going to pass.",
    )
    ledger = _write_ledger(
        tmp_path / "labels.json",
        [
            _audit(
                slug,
                status="Out",
                evidence_quote="I'm going to pass.",
                evidence_turn_index=3,
            )
        ],
    )

    records = build_precedent_corpus(
        source, ledger, tmp_path / "out", ("Jillian", "Jillian Manus")
    )

    assert [
        evidence.model_dump() for evidence in records[0].decision.evidence
    ] == [
        {
            "turn_start": 3,
            "turn_end": 3,
            "text": "I'm going to pass.",
        }
    ]


def test_builder_rejects_observed_label_without_matching_source_span(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    _write_source(source, "18-example", "Charles: No.")
    ledger = _write_ledger(
        tmp_path / "labels.json",
        [_audit("18-example", evidence_quote="I am investing.")],
    )

    with pytest.raises(ValueError, match="evidence quote not found"):
        build_precedent_corpus(source, ledger, tmp_path / "out", ("Charles",))
    assert not (tmp_path / "out" / "corpus-manifest.json").exists()


def test_failed_rebuild_removes_an_existing_manifest(tmp_path: Path) -> None:
    source = tmp_path / "source"
    slug = "18-example"
    _write_source(source, slug, "Charles: I think I'm in.")
    ledger = _write_ledger(tmp_path / "labels.json", [_audit(slug)])
    output = tmp_path / "out"
    build_precedent_corpus(source, ledger, output, ("Charles",))
    assert (output / "corpus-manifest.json").is_file()
    _write_ledger(ledger, [_audit(slug, status="invalid")])

    with pytest.raises(ValueError, match="pitch_window_decision"):
        build_precedent_corpus(source, ledger, output, ("Charles",))

    assert not (output / "corpus-manifest.json").exists()


def test_builder_rejects_output_source_symlink_escape(tmp_path: Path) -> None:
    source = tmp_path / "source"
    slug = "18-example"
    _write_source(source, slug, "Charles: I think I'm in.")
    ledger = _write_ledger(tmp_path / "labels.json", [_audit(slug)])
    output = tmp_path / "out"
    outside = tmp_path / "outside"
    output.mkdir()
    outside.mkdir()
    (output / "sources").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="within corpus output"):
        build_precedent_corpus(source, ledger, output, ("Charles",))

    assert not (outside / f"{slug}.json").exists()
    assert not (output / "corpus-manifest.json").exists()


def test_missing_or_null_transcript_cannot_be_observed(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write_source(source, "18-example", None)
    ledger = _write_ledger(tmp_path / "labels.json", [_audit("18-example")])

    with pytest.raises(ValueError, match="evidence quote not found"):
        build_precedent_corpus(source, ledger, tmp_path / "out", ("Charles",))


@pytest.mark.parametrize("transcript", [None, ""])
def test_unobserved_empty_transcript_rejects_unknown_context(
    tmp_path: Path, transcript: str | None
) -> None:
    source = tmp_path / "source"
    slug = "99-example"
    _write_source(source, slug, transcript)
    ledger = _write_ledger(
        tmp_path / "labels.json",
        [
            _audit(
                slug,
                status="Unobserved",
                decision_context="unexpected",
                evidence_quote="words that must be ignored",
            )
        ],
    )

    with pytest.raises(ValueError, match="decision_context"):
        build_precedent_corpus(
            source, ledger, tmp_path / "out", ("CHARLES",)
        )
    assert not (tmp_path / "out" / "corpus-manifest.json").exists()


def test_builder_preserves_same_session_reversal_context(tmp_path: Path) -> None:
    source = tmp_path / "source"
    slug = "8-example"
    _write_source(source, slug, "Jillian: Fine, Jillian's giving a 25.")
    ledger = _write_ledger(
        tmp_path / "labels.json",
        [
            _audit(
                slug,
                decision_context="same_session_reversal",
                evidence_quote="Fine, Jillian's giving a 25.",
                evidence_turn_index=0,
            )
        ],
    )

    records = build_precedent_corpus(
        source, ledger, tmp_path / "out", ("Jillian", "Jillian Manus")
    )

    assert records[0].decision.context == "same_session_reversal"


@pytest.mark.parametrize(
    "invalid_pointer",
    (0, 99, True, "1"),
)
def test_builder_rejects_invalid_present_evidence_pointer(
    tmp_path: Path, invalid_pointer: object
) -> None:
    source = tmp_path / "source"
    slug = "18-example"
    _write_source(
        source,
        slug,
        "Founder: Opening pitch.\nCharles: I think I'm in.",
    )
    ledger = _write_ledger(
        tmp_path / "labels.json",
        [_audit(slug, evidence_turn_index=invalid_pointer)],
    )

    with pytest.raises(ValueError, match="evidence_turn_index"):
        build_precedent_corpus(source, ledger, tmp_path / "out", ("Charles",))


def test_builder_rejects_stale_quote_at_present_evidence_pointer(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    slug = "18-example"
    _write_source(source, slug, "Founder: Opening.\nCharles: I am out.")
    ledger = _write_ledger(
        tmp_path / "labels.json",
        [
            _audit(
                slug,
                evidence_quote="I think I'm in.",
                evidence_turn_index=1,
            )
        ],
    )

    with pytest.raises(ValueError, match="evidence_turn_index"):
        build_precedent_corpus(source, ledger, tmp_path / "out", ("Charles",))


def test_builder_never_absorbs_a_following_different_speaker_for_evidence(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    slug = "18-example"
    _write_source(
        source,
        slug,
        "Charles: I think\nFounder: I'm in.",
    )
    ledger = _write_ledger(
        tmp_path / "labels.json",
        [
            _audit(
                slug,
                evidence_quote="I think I'm in.",
                evidence_turn_index=0,
            )
        ],
    )

    with pytest.raises(ValueError, match="evidence_turn_index"):
        build_precedent_corpus(source, ledger, tmp_path / "out", ("Charles",))


def test_builder_rejects_ambiguous_alias_only_fallback_matches(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    slug = "18-example"
    _write_source(
        source,
        slug,
        "Charles: I think I'm in.\nFounder: Thanks.\nCharles: I think I'm in.",
    )
    ledger = _write_ledger(tmp_path / "labels.json", [_audit(slug)])

    with pytest.raises(ValueError, match="ambiguous"):
        build_precedent_corpus(source, ledger, tmp_path / "out", ("Charles",))


def test_duplicate_ledger_slugs_are_rejected(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write_source(source, "18-example", "Charles: No.")
    ledger = _write_ledger(
        tmp_path / "labels.json",
        [_audit("18-example"), _audit("18-example")],
    )

    with pytest.raises(ValueError, match="duplicate episode slugs"):
        build_precedent_corpus(source, ledger, tmp_path / "out", ("Charles",))


@pytest.mark.parametrize(
    ("status_present", "invalid_status"),
    [
        pytest.param(False, None, id="missing"),
        pytest.param(True, None, id="null"),
        pytest.param(True, "IN", id="wrong-case"),
        pytest.param(True, "Ambiguous", id="ambiguous"),
        pytest.param(True, "Maybe", id="arbitrary"),
    ],
)
def test_invalid_ledger_decision_status_is_rejected_before_manifest_publish(
    tmp_path: Path, status_present: bool, invalid_status: str | None
) -> None:
    source = tmp_path / "source"
    slug = "18-example"
    _write_source(source, slug, "Charles: I think I'm in.")
    row = _audit(slug)
    if status_present:
        row["pitch_window_decision"] = invalid_status
    else:
        row.pop("pitch_window_decision")
    ledger = _write_ledger(tmp_path / "labels.json", [row])
    output = tmp_path / "out"

    with pytest.raises(ValueError, match="pitch_window_decision"):
        build_precedent_corpus(source, ledger, output, ("Charles",))
    assert not (output / "corpus-manifest.json").exists()


def test_missing_ledger_source_is_rejected_even_if_other_transcripts_exist(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    _write_source(source, "20-unlisted", "Charles: I think I'm in.")
    ledger = _write_ledger(tmp_path / "labels.json", [_audit("18-missing")])

    with pytest.raises(ValueError, match="source transcript record missing: 18-missing"):
        build_precedent_corpus(source, ledger, tmp_path / "out", ("Charles",))


def test_builder_accepts_nonnumeric_gnara_style_slug(tmp_path: Path) -> None:
    source = tmp_path / "source"
    slug = "gnara-disrupting-your-pants"
    _write_source(source, slug, None)
    ledger = _write_ledger(
        tmp_path / "labels.json",
        [_audit(slug, status="Unobserved", decision_context="unclear")],
    )

    records = build_precedent_corpus(
        source, ledger, tmp_path / "out", ("Charles",)
    )

    assert records[0].episode_slug == slug
    assert records[0].episode_number is None
    assert records[0].decision.status == "unobserved"


def test_parser_preserves_narration_and_exact_multiline_turn_text() -> None:
    turns = parse_speaker_turns(
        "Opening  narration.\nCharles:  Keep  both spaces.\nContinuation: is a speaker\n trailing continuation"
    )

    assert [turn.model_dump() for turn in turns] == [
        {"turn_index": 0, "speaker": "Narrator", "text": "Opening  narration."},
        {"turn_index": 1, "speaker": "Charles", "text": " Keep  both spaces."},
        {"turn_index": 2, "speaker": "Continuation", "text": "is a speaker\n trailing continuation"},
    ]


def test_parser_recognizes_speaker_name_split_from_colon() -> None:
    turns = parse_speaker_turns(
        "Founder:\nFirst statement.\nFounder\n: Second statement.\n"
        "Elizabeth\n: What is revenue?"
    )

    assert [turn.model_dump() for turn in turns] == [
        {"turn_index": 0, "speaker": "Founder", "text": "First statement."},
        {"turn_index": 1, "speaker": "Founder", "text": "Second statement."},
        {"turn_index": 2, "speaker": "Elizabeth", "text": "What is revenue?"},
    ]


def test_parser_recognizes_unicode_speaker_names_for_cross_investor_sources() -> None:
    turns = parse_speaker_turns(
        "Iñaki: We reduce corporate energy costs.\n"
        "Elizabeth: How much can customers save?"
    )

    assert [turn.model_dump() for turn in turns] == [
        {
            "turn_index": 0,
            "speaker": "Iñaki",
            "text": "We reduce corporate energy costs.",
        },
        {
            "turn_index": 1,
            "speaker": "Elizabeth",
            "text": "How much can customers save?",
        },
    ]


def test_manifest_and_chunks_are_deterministic_and_hash_bound(tmp_path: Path) -> None:
    source = tmp_path / "source"
    slug = "18-example"
    transcript = "\n".join(
        f"{'Charles' if index == 8 else 'Founder'}: line {index}"
        for index in range(9)
    )
    source_path = _write_source(source, slug, transcript)
    ledger = _write_ledger(
        tmp_path / "labels.json",
        [_audit(slug, evidence_quote="line 8")],
    )
    output = tmp_path / "out"

    build_precedent_corpus(source, ledger, output, ("charles",))
    first = {
        path.relative_to(output).as_posix(): path.read_bytes()
        for path in output.rglob("*")
        if path.is_file()
    }
    build_precedent_corpus(source, ledger, output, ("charles",))
    second = {
        path.relative_to(output).as_posix(): path.read_bytes()
        for path in output.rglob("*")
        if path.is_file()
    }

    assert first == second
    manifest = json.loads(first["corpus-manifest.json"])
    record_bytes = first[f"records/{slug}.json"]
    assert first[f"sources/{slug}.json"] == source_path.read_bytes()
    record = json.loads(record_bytes)
    assert record["source_path"] == f"sources/{slug}.json"
    assert manifest == {
        "records": [
            {
                "decision_status": "In",
                "episode_slug": slug,
                "record_path": f"records/{slug}.json",
                "record_sha256": sha256(record_bytes).hexdigest(),
                "source_path": f"sources/{slug}.json",
                "source_sha256": sha256(source_path.read_bytes()).hexdigest(),
            }
        ],
        "schema": "precedent-corpus-v1",
    }
    chunks = [json.loads(line) for line in first["chunks.jsonl"].splitlines()]
    assert chunks == [
        {
            "chunk_id": f"H-{sha256(f'{slug}:0'.encode()).hexdigest()[:20]}",
            "episode_slug": slug,
            "source_sha256": sha256(source_path.read_bytes()).hexdigest(),
            "text": "\n".join(f"Founder: line {index}" for index in range(8)),
            "turn_end": 7,
            "turn_start": 0,
        },
        {
            "chunk_id": f"H-{sha256(f'{slug}:8'.encode()).hexdigest()[:20]}",
            "episode_slug": slug,
            "source_sha256": sha256(source_path.read_bytes()).hexdigest(),
            "text": "Charles: line 8",
            "turn_end": 8,
            "turn_start": 8,
        },
    ]


def test_cli_requires_arguments_and_accepts_repeatable_aliases(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit, match="2"):
        main([])

    source = tmp_path / "source"
    _write_source(source, "18-example", "Charles Hudson: No.")
    ledger = _write_ledger(
        tmp_path / "labels.json",
        [_audit("18-example", status="Out", evidence_quote="No.")],
    )
    output = tmp_path / "out"

    assert (
        main(
            [
                "--transcripts",
                str(source),
                "--ledger",
                str(ledger),
                "--output",
                str(output),
                "--investor-alias",
                "Charles",
                "--investor-alias",
                "Charles Hudson",
            ]
        )
        == 0
    )
    assert "1 precedent records" in capsys.readouterr().out
    row = json.loads((output / "records" / "18-example.json").read_text())
    assert row["investor_aliases"] == ["Charles", "Charles Hudson"]
    assert row["investor_present"] is True


def test_audited_charles_corpus_is_complete_exact_and_self_contained() -> None:
    ledger_path = ROOT / "evaluation/labels/charles_pitch_window_decisions.json"
    corpus_root = (
        ROOT
        / "inputs/data/investors/charles-hudson-precursor-ventures/precedents"
    )
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    manifest = json.loads(
        (corpus_root / "corpus-manifest.json").read_text(encoding="utf-8")
    )

    assert len(ledger) == len(manifest["records"]) == 113
    assert {
        status: sum(row["pitch_window_decision"] == status for row in ledger)
        for status in ("In", "Out", "Unobserved")
    } == {"In": 19, "Out": 67, "Unobserved": 27}
    assert sum(
        1
        for line in (corpus_root / "chunks.jsonl").read_text().splitlines()
        if line
    ) == 2515

    validated_files = validate_precedent_corpus(corpus_root)
    assert len(validated_files) == 228
    corpus = PrecedentCorpus.build(corpus_root)
    assert len(corpus.list_episodes()) == 113
    records = {
        summary.episode_slug: json.loads(
            (corpus_root / f"records/{summary.episode_slug}.json").read_text()
        )
        for summary in corpus.list_episodes()
    }

    for slug in (
        "41-can-this-startup-help-retailers-take-on-amazon",
        "127-nectir-the-classroom-of-the-future",
    ):
        assert records[slug]["decision"]["status"] == "In"
        assert records[slug]["decision"]["conditions"] == [
            "portfolio de-confliction"
        ]
    nectir = records["127-nectir-the-classroom-of-the-future"]
    assert "50 to 100k" in nectir["decision"]["evidence"][0]["text"]
    assert records["gnara-disrupting-your-pants"]["episode_number"] is None

    for record in records.values():
        source = corpus_root / record["source_path"]
        assert sha256(source.read_bytes()).hexdigest() == record["source_sha256"]
        if record["decision"]["status"] != "unobserved":
            assert record["decision"]["evidence"]
            for evidence in record["decision"]["evidence"]:
                exact = "\n".join(
                    turn["text"]
                    for turn in record["turns"][
                        evidence["turn_start"] : evidence["turn_end"] + 1
                    ]
                )
                assert evidence["text"] == exact
