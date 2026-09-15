from __future__ import annotations

from pathlib import Path
import json

from vc_clone_graph.evaluation import load_registry
from vc_clone_graph.phase1_references import (
    ReferenceIdentity,
    build_reference_corpus,
    normalize_reference,
    summarize_extraction_usage,
)


def test_normalizes_legacy_reference_to_common_phase1_schema() -> None:
    normalized = normalize_reference(
        {
            "episode_slug": "1-one",
            "vc_slug": "legacy-vc",
            "decision": "In",
            "rationales": [
                {
                    "label": "founder_qualities",
                    "direction": "positive",
                    "salience": "primary",
                    "confidence": 0.8,
                    "evidence_span": ["I really like this founder."],
                }
            ],
        },
        source_path=Path("legacy.json"),
        source_format="legacy-reference-rationales",
        vc_slug="example-vc",
        vc_name="Example VC",
        actual_decision="In",
        taxonomy_labels={"founder_qualities"},
    )

    assert normalized["schema"] == "phase1-rationale-reference-v1"
    assert normalized["reference_kind"] == "automated_transcript_observed_candidate"
    assert normalized["human_validated"] is False
    assert normalized["rationales"] == [
        {
            "rationale_label": "founder_qualities",
            "direction": "positive",
            "salience": "primary",
            "confidence": 0.8,
            "activation": "evaluated",
            "utterance_type": "assessment",
            "decision_link": "unspecified",
            "evidence": ["I really like this founder."],
            "source_rationale_id": None,
        }
    ]


def test_normalizes_observed_reference_by_resolving_evidence_ids() -> None:
    normalized = normalize_reference(
        {
            "schema": "transcript-observed-rationales-v1",
            "episode_slug": "1-one",
            "observed_decision": "Out",
            "evidence_registry": [
                {
                    "evidence_id": "E1",
                    "turn_index": 4,
                    "speaker": "Example",
                    "text": "The market is too small.",
                }
            ],
            "rationales": [
                {
                    "rationale_id": "R1",
                    "rationale_label": "market_size_assessment",
                    "utterance_type": "decision_reason",
                    "activation": "evaluated",
                    "direction": "negative",
                    "salience": "primary",
                    "confidence": 0.9,
                    "evidence_ids": ["E1"],
                    "explanation": "Small market.",
                    "decision_link": "explicit",
                }
            ],
        },
        source_path=Path("observed.json"),
        source_format="transcript-observed-rationales-v1",
        vc_slug="example-vc",
        vc_name="Example VC",
        actual_decision="Out",
        taxonomy_labels={"market_size_assessment"},
    )

    assert normalized["rationales"][0]["evidence"] == ["The market is too small."]
    assert normalized["rationales"][0]["source_rationale_id"] == "R1"
    assert normalized["source_format"] == "transcript-observed-rationales-v1"


def test_summarizes_all_extraction_attempt_usage(tmp_path: Path) -> None:
    for index, usage in enumerate(
        (
            {"input_tokens": 100, "output_tokens": 20, "reasoning_output_tokens": 7},
            {"input_tokens": 150, "output_tokens": 30, "reasoning_output_tokens": 11},
        )
    ):
        destination = tmp_path / f"attempt-{index}" / "episode" / "usage.json"
        destination.parent.mkdir(parents=True)
        destination.write_text(json.dumps(usage))

    assert summarize_extraction_usage(tmp_path) == {
        "model_call_count": 2,
        "input_tokens": 250,
        "output_tokens": 50,
        "reasoning_output_tokens": 18,
    }


def test_corpus_builder_prefers_rich_reference_and_reports_missing(tmp_path: Path) -> None:
    registry_path = tmp_path / "evaluation" / "registry.json"
    labels_path = tmp_path / "evaluation" / "labels.json"
    labels_path.parent.mkdir(parents=True)
    labels_path.write_text(
        json.dumps(
            [
                {
                    "episode_slug": slug,
                    "pitch_window_decision": decision,
                    "evaluation_eligible": True,
                }
                for slug, decision in (
                    ("1-one", "In"),
                    ("2-two", "Out"),
                    ("3-three", "Out"),
                )
            ]
        )
    )
    registry_path.write_text(
        json.dumps(
            {
                "schema": "canonical-vc-evaluation-registry-v1",
                "investors": {
                    "example-vc": {
                        "display_name": "Example VC",
                        "label_file": "labels.json",
                        "eligible_count": 3,
                        "sources": [{"kind": "summary_files", "paths": ["unused.json"]}],
                    }
                },
            }
        )
    )
    agentic = tmp_path / "agentic"
    taxonomy = agentic / "taxonomy" / "codebook_v_final.json"
    taxonomy.parent.mkdir(parents=True)
    taxonomy.write_text(json.dumps([{"label": "founder_qualities"}]))
    legacy = agentic / "data" / "reference_rationales" / "1-one__legacy-vc.json"
    legacy.parent.mkdir(parents=True)
    legacy.write_text(
        json.dumps(
            {
                "episode_slug": "1-one",
                "vc_slug": "legacy-vc",
                "decision": "In",
                "rationales": [
                    {
                        "label": "founder_qualities",
                        "direction": "positive",
                        "salience": "secondary",
                        "confidence": 0.6,
                        "evidence_span": ["Legacy evidence."],
                    }
                ],
            }
        )
    )
    rich = agentic / "outputs" / "old" / "1-one" / "reference.json"
    rich.parent.mkdir(parents=True)
    rich.write_text(
        json.dumps(
            {
                "schema": "transcript-observed-rationales-v1",
                "episode_slug": "1-one",
                "observed_decision": "In",
                "evidence_registry": [
                    {
                        "evidence_id": "E1",
                        "turn_index": 1,
                        "speaker": "Example",
                        "text": "Rich evidence.",
                    }
                ],
                "rationales": [
                    {
                        "rationale_id": "R1",
                        "rationale_label": "founder_qualities",
                        "utterance_type": "assessment",
                        "activation": "evaluated",
                        "direction": "positive",
                        "salience": "primary",
                        "confidence": 0.9,
                        "evidence_ids": ["E1"],
                        "decision_link": "supporting",
                    }
                ],
            }
        )
    )
    fallback_legacy = agentic / "data" / "reference_rationales" / "2-two__legacy-vc.json"
    fallback_legacy.write_text(
        json.dumps(
            {
                "episode_slug": "2-two",
                "vc_slug": "legacy-vc",
                "decision": "Out",
                "rationales": [
                    {
                        "label": "founder_qualities",
                        "direction": "negative",
                        "salience": "primary",
                        "confidence": 0.8,
                        "evidence_span": ["Audited-label-consistent legacy evidence."],
                    }
                ],
            }
        )
    )
    conflicting_rich = agentic / "outputs" / "old" / "2-two" / "reference.json"
    conflicting_rich.parent.mkdir(parents=True)
    conflicting_rich.write_text(
        json.dumps(
            {
                "schema": "transcript-observed-rationales-v1",
                "episode_slug": "2-two",
                "observed_decision": "In",
                "evidence_registry": [
                    {
                        "evidence_id": "E1",
                        "turn_index": 1,
                        "speaker": "Example",
                        "text": "Superseded decision evidence.",
                    }
                ],
                "rationales": [
                    {
                        "rationale_id": "R1",
                        "rationale_label": "founder_qualities",
                        "utterance_type": "assessment",
                        "activation": "evaluated",
                        "direction": "positive",
                        "salience": "primary",
                        "confidence": 0.8,
                        "evidence_ids": ["E1"],
                        "decision_link": "supporting",
                    }
                ],
            }
        )
    )
    output = tmp_path / "corpus"

    manifest = build_reference_corpus(
        registry=load_registry(registry_path),
        identities={
            "example-vc": ReferenceIdentity(
                vc_slug="example-vc",
                vc_name="Example VC",
                rich_speaker="Example",
                legacy_suffix="legacy-vc",
            )
        },
        agentic_root=agentic,
        output_root=output,
        allow_missing=True,
    )

    record = json.loads((output / "records" / "example-vc" / "1-one.json").read_text())
    assert record["source_format"] == "transcript-observed-rationales-v1"
    assert record["source_tier"] == "previously_validated"
    assert record["source_origin_path"] == str(rich)
    assert record["rationales"][0]["evidence"] == ["Rich evidence."]
    fallback_record = json.loads((output / "records" / "example-vc" / "2-two.json").read_text())
    assert fallback_record["source_format"] == "legacy-reference-rationales"
    assert fallback_record["source_tier"] == "legacy_fallback"
    assert manifest["record_count"] == 2
    assert manifest["by_vc"]["example-vc"]["source_tier_counts"] == {
        "legacy_fallback": 1,
        "previously_validated": 1,
    }
    assert manifest["missing"] == {"example-vc": ["3-three"]}
