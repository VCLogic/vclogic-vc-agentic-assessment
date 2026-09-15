from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import importlib.util
import json
import os
import sys

import numpy as np
import pytest

import vc_clone_graph.standalone_pitch_models as standalone
from vc_clone_graph.artifacts import freeze_model
from vc_clone_graph.posthoc_semantic_models import SemanticCalibrationRecord
from vc_clone_graph.schemas_v4 import InvestigationV4
from vc_clone_graph.standalone_pitch_models import (
    StandalonePitchRecord,
    load_verified_pitch_population,
    nested_standalone_predictions,
    write_standalone_analysis,
)


VC = "charles-hudson-precursor-ventures"


def _investigation(slug: str) -> InvestigationV4:
    return InvestigationV4.model_validate(
        {
            "schema_version": "investigation-v4",
            "episode_slug": slug,
            "questions": [
                {
                    "question": "Can this team execute?",
                    "status": "partial",
                    "answer": "The launch is real, while retention is unresolved.",
                    "evidence_ids": ["W-founder"],
                }
            ],
            "rationales": [
                {
                    "rationale_id": "R1",
                    "taxonomy_label": "founder_execution",
                    "direction": "positive",
                    "salience": "primary",
                    "confidence": 0.82,
                    "pitch_evidence": ["We shipped the product."],
                    "pitch_evidence_ids": ["P-001"],
                    "wiki_evidence_ids": ["W-founder"],
                    "historical_evidence_ids": ["H-prior"],
                    "justification": "The founder demonstrated execution.",
                },
                {
                    "rationale_id": "R2",
                    "taxonomy_label": "market_size_assessment",
                    "direction": "negative",
                    "salience": "secondary",
                    "confidence": 0.61,
                    "pitch_evidence": ["The initial segment is narrow."],
                    "pitch_evidence_ids": ["P-002"],
                    "wiki_evidence_ids": ["W-market"],
                    "historical_evidence_ids": [],
                    "justification": "Venture scale remains uncertain.",
                },
            ],
            "conflicts": ["Execution is positive but scale is unresolved."],
            "unmapped_observations": [],
            "information_sufficient": True,
            "sufficiency_assessment": "Enough information exists.",
            "searchable_questions": [],
            "diligence_questions": ["What is retention?"],
            "next_search_objectives": [],
            "summary": "Execution is positive and scale is uncertain.",
        }
    )


def _semantic_record(root: Path, slug: str, target: int) -> SemanticCalibrationRecord:
    _, digest = freeze_model(root / "phase1", "investigation", _investigation(slug))
    return SemanticCalibrationRecord(
        episode_slug=slug,
        target=target,
        target_decision="In" if target else "Out",
        original_decision="Out",
        original_likelihood=0.3,
        review_priority_score=0.4,
        phase1_text="unused",
        full_text="unused",
        structured_features={"decision_path__out": 1.0},
        artifact_root=str(root),
        phase1_sha256=digest,
        phase2_sha256="f" * 64,
    )


def test_loads_firewall_verified_pitch_and_phase1_features(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    run_roots = [tmp_path / "runs/1-first", tmp_path / "runs/2-second"]
    records = [
        _semantic_record(run_roots[0], "1-first", 1),
        _semantic_record(run_roots[1], "2-second", 0),
    ]
    pitch_paths = {}
    for record in records:
        pitch = tmp_path / "pitches" / f"{record.episode_slug}.txt"
        pitch.parent.mkdir(parents=True, exist_ok=True)
        pitch.write_text(f"Founder pitch for {record.episode_slug}", encoding="utf-8")
        pitch_paths[record.episode_slug] = pitch

    monkeypatch.setattr(standalone, "load_semantic_population", lambda *_: records)
    calls = []

    def fake_verify(root: Path, vc_slug: str, episode_slug: str):
        calls.append((root, vc_slug, episode_slug))
        return SimpleNamespace(pitch=pitch_paths[episode_slug])

    monkeypatch.setattr(standalone, "verify_package", fake_verify)

    loaded = load_verified_pitch_population(
        tmp_path, tmp_path / "status.json", tmp_path / "labels.json", VC
    )

    assert [row.pitch_text for row in loaded] == [
        "Founder pitch for 1-first",
        "Founder pitch for 2-second",
    ]
    assert calls == [
        (tmp_path, VC, "1-first"),
        (tmp_path, VC, "2-second"),
    ]
    first = loaded[0].rationale_features
    assert first["label__founder_execution__signed_confidence"] == pytest.approx(0.82)
    assert first["label__market_size_assessment__signed_confidence"] == pytest.approx(-0.61)
    assert first["positive_primary_confidence_sum"] == pytest.approx(0.82)
    assert first["negative_secondary_confidence_sum"] == pytest.approx(0.61)
    assert first["wiki_evidence_count"] == 2.0
    assert first["historical_evidence_count"] == 1.0
    assert first["conflict_count"] == 1.0
    assert not any("decision" in name or "likelihood" in name for name in first)


def _standalone_records(*, flip_first: bool = False) -> list[StandalonePitchRecord]:
    records = []
    for index in range(18):
        target = int(index % 3 == 0)
        if flip_first and index == 0:
            target = 1 - target
        signal = 1.0 if index % 3 == 0 else -1.0
        records.append(
            StandalonePitchRecord(
                episode_slug=f"{index + 1}-example",
                target=target,
                target_decision="In" if target else "Out",
                original_decision="In" if index % 4 == 0 else "Out",
                original_likelihood=0.5 + 0.1 * signal,
                pitch_text=(
                    "strong founder with traction "
                    if signal > 0
                    else "weak market and economics "
                )
                + f"case {index % 2}",
                rationale_features={
                    "positive_primary_confidence_sum": max(signal, 0.0),
                    "negative_primary_confidence_sum": max(-signal, 0.0),
                    "label__founder_execution__signed_confidence": signal,
                },
                pitch_sha256=str(index).zfill(64),
                phase1_sha256=str(index + 1).zfill(64),
            )
        )
    return records


def _pitch_embeddings(records: list[StandalonePitchRecord]) -> np.ndarray:
    return np.asarray(
        [
            [1.0, 0.0] if "strong founder" in record.pitch_text else [0.0, 1.0]
            for record in records
        ],
        dtype=float,
    )


def test_nested_standalone_methods_exclude_held_episode_and_label() -> None:
    records = _standalone_records()
    embeddings = _pitch_embeddings(records)
    methods = (
        "rationale_only",
        "pitch_tfidf_only",
        "rationale_plus_pitch_tfidf",
        "pitch_embedding_only",
        "rationale_plus_pitch_embedding",
    )

    original = nested_standalone_predictions(records, embeddings)
    flipped = nested_standalone_predictions(
        _standalone_records(flip_first=True), embeddings, methods=("rationale_only",)
    )

    assert set(original) == set(methods)
    for rows in original.values():
        assert len(rows) == len(records)
        assert all(
            row.episode_slug not in row.training_slugs
            and len(row.training_slugs) == len(records) - 1
            for row in rows
        )
    assert flipped["rationale_only"][0].score == pytest.approx(
        original["rationale_only"][0].score
    )
    assert flipped["rationale_only"][0].threshold == pytest.approx(
        original["rationale_only"][0].threshold
    )


def test_write_standalone_analysis(tmp_path: Path) -> None:
    records = _standalone_records()
    predictions = nested_standalone_predictions(records, _pitch_embeddings(records))

    result = write_standalone_analysis(
        records,
        predictions,
        tmp_path / "analysis",
        embedding_metadata={"model": "test-model", "revision": "abc123"},
        source_manifest={"status_path": "status.json", "input_root": "inputs"},
    )

    assert {path.name for path in (tmp_path / "analysis").iterdir()} == {
        "embedding-metadata.json",
        "manifest.json",
        "metrics.json",
        "predictions.csv",
        "rankings.csv",
        "report.md",
    }
    manifest = json.loads(
        (tmp_path / "analysis/manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["api_cost_usd"] == 0.0
    assert manifest["prediction_reruns"] == 0
    assert manifest["pipeline_modified"] is False
    assert manifest["population"] == {"episodes": 18, "ins": 6, "outs": 12}
    assert set(result["methods"]) == {
        "raw_v4",
        "rationale_only",
        "pitch_tfidf_only",
        "rationale_plus_pitch_tfidf",
        "pitch_embedding_only",
        "rationale_plus_pitch_embedding",
    }


def test_cli_arguments_and_offline_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    script = Path(__file__).parents[1] / "scripts/analyze_rationale_pitch_standalone.py"
    spec = importlib.util.spec_from_file_location("standalone_cli", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(script),
            "--input-root",
            "inputs",
            "--vc-slug",
            VC,
            "--status",
            "status.json",
            "--labels",
            "labels.json",
            "--output",
            "output",
            "--embedding-model",
            "nomic-ai/nomic-embed-text-v1.5",
            "--embedding-revision",
            "abc123",
        ],
    )

    args = module.parse_args()
    module.configure_offline_environment()

    assert args.input_root == Path("inputs")
    assert args.vc_slug == VC
    assert args.batch_size == 16
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert os.environ["TRANSFORMERS_OFFLINE"] == "1"
    assert os.environ["TOKENIZERS_PARALLELISM"] == "false"
