from __future__ import annotations

import csv
import hashlib
import json

from vc_clone_graph.rationale_association_annotations import (
    AssociationAnnotationEvaluationCase,
    RationaleAssociationAnnotations,
    write_rationale_association_analysis,
)


def _annotations(slug: str, associated: bool) -> RationaleAssociationAnnotations:
    candidate = (
        [
            {
                "taxonomy_label": "market_size_assessment",
                "posterior_probability": 0.75,
                "support": 4,
                "lift": 2.0,
                "antecedent_labels": ["founder_execution"],
                "status": "hypothesis_only",
            }
        ]
        if associated
        else []
    )
    return RationaleAssociationAnnotations.model_validate(
        {
            "schema": "rationale-association-annotations-v1",
            "scientific_status": "fold_safe_advisory_hypotheses",
            "vc_slug": "charles-hudson",
            "episode_slug": slug,
            "investigation_sha256": hashlib.sha256(slug.encode()).hexdigest(),
            "thresholds": {
                "min_posterior_probability": 0.6,
                "min_support": 3,
                "min_lift": 1.0,
                "max_per_source": 5,
            },
            "training_episode_slugs": ["other"],
            "claim_annotations": [
                {
                    "rationale_id": "R1",
                    "source_taxonomy_label": "founder_execution",
                    "associated_rationales": candidate,
                }
            ],
            "material_statement_annotations": [
                {
                    "pitch_evidence_id": "P-001",
                    "mapped_ids": ["R1"],
                    "source_taxonomy_labels": ["founder_execution"],
                    "associated_rationales": candidate,
                }
            ],
            "unmapped_observation_annotations": [],
            "episode_level_associations": candidate,
        }
    )


def test_writer_emits_sidecars_metrics_manifest_and_zero_cost_report(tmp_path) -> None:
    source = tmp_path / "source.json"
    source.write_text('{"source": true}\n', encoding="utf-8")
    cases = [
        AssociationAnnotationEvaluationCase(
            annotations=_annotations("episode-a", True),
            observed_labels=("founder_execution",),
            reference_labels=("founder_execution", "market_size_assessment"),
        ),
        AssociationAnnotationEvaluationCase(
            annotations=_annotations("episode-b", False),
            observed_labels=("founder_execution",),
            reference_labels=("founder_execution",),
        ),
    ]

    paths = write_rationale_association_analysis(
        cases,
        tmp_path / "report",
        input_paths={"source": source},
    )

    metrics = json.loads(paths["metrics"].read_text(encoding="utf-8"))
    assert metrics["api_cost_usd"] == 0.0
    assert metrics["case_count"] == 2
    assert metrics["missing_reference_count"] == 1
    assert metrics["missing_reference_recall"] == 1.0
    assert metrics["hypothesis_precision"] == 1.0
    assert metrics["claim_association_coverage"] == 0.5
    assert metrics["material_statement_association_coverage"] == 0.5
    assert (tmp_path / "report/records/charles-hudson/episode-a.json").is_file()
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    assert manifest["inputs"]["source"]["sha256"] == hashlib.sha256(
        source.read_bytes()
    ).hexdigest()
    assert manifest["record_count"] == 2
    with paths["annotations"].open(encoding="utf-8", newline="") as stream:
        assert len(list(csv.DictReader(stream))) == 2
    assert "automated transcript-derived" in paths["evaluation"].read_text(
        encoding="utf-8"
    )
