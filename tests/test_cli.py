from pathlib import Path
from dataclasses import replace
from hashlib import sha256
import json
import shutil
from types import SimpleNamespace

import pytest

import vc_clone_graph.cli as cli_module
from vc_clone_graph.artifacts import canonical_bytes, verify_phase1_artifacts
from vc_clone_graph.cli import _generation_provider, _provider, main
from vc_clone_graph.config import RunConfig
from vc_clone_graph.providers.base import CompositeProvider
from vc_clone_graph.providers.fake import FakeProvider
from vc_clone_graph.providers.ollama import OllamaProvider
from vc_clone_graph.providers.openai import OpenAIProvider
from vc_clone_graph.providers.openrouter import OpenRouterProvider
from vc_clone_graph.precedent_builder import build_precedent_corpus
from vc_clone_graph.retrieval import HybridWikiIndex
from vc_clone_graph.schemas import InvestigationV3
from vc_clone_graph.workflow_v44 import VCDecisionWorkflowV44
from tests.test_graph import (
    as_decision_v3,
    decision,
    phase1,
    phase1_v3,
    phase2_plan,
)
from tests.test_workflow_v4 import (
    ScriptedProvider as ScriptedProviderV4,
    decision as decision_v4,
    investigation as investigation_v4,
    plan as plan_v4,
)


ROOT = Path(__file__).resolve().parents[1]


class _LocalCliV44Embedder:
    metadata = {
        "backend": "sentence_transformers",
        "model": "cli-v44-local",
        "revision": "pinned-revision",
        "normalize": True,
        "document_prefix": "search_document: ",
        "query_prefix": "search_query: ",
    }

    def embed(self, texts):
        return [
            [1.0, float("founder" in text.casefold()), float(len(text))]
            for text in texts
        ]

    def embed_documents(self, texts):
        return self.embed(texts)

    def embed_queries(self, texts):
        return self.embed(texts)


class _QueryOnlyCliV44Embedder(_LocalCliV44Embedder):
    def embed_documents(self, texts):
        raise RuntimeError("taxonomy document embeddings unavailable")


def _minimal_v44_package(
    root: Path,
    *,
    vc_slug: str = "elizabeth-yin-hustle-fund",
    episode_slug: str = "135-thoras-ai-the-twin-effect",
    pitch_bytes: bytes = b"Founder reports five paid pilots.\n",
) -> SimpleNamespace:
    inputs = root / "inputs"
    wiki = inputs / "wiki" / vc_slug
    wiki.mkdir(parents=True)
    (wiki / "principles.md").write_text(
        "# Founder execution\nPaid pilots are useful early execution evidence.\n",
        encoding="utf-8",
    )
    pitch = inputs / "data/investors" / vc_slug / "pitches" / f"{episode_slug}.txt"
    pitch.parent.mkdir(parents=True)
    pitch.write_bytes(pitch_bytes)
    registry = inputs / "investors" / f"{vc_slug}.toml"
    registry.parent.mkdir(parents=True)
    registry.write_text(
        f'vc_slug = "{vc_slug}"\n'
        f'wiki_path = "wiki/{vc_slug}"\n'
        'display_name = "Elizabeth Yin"\n'
        'check_tiers = ["small_exploratory"]\n',
        encoding="utf-8",
    )
    taxonomy = inputs / "taxonomy/codebook_v_final.json"
    taxonomy.parent.mkdir(parents=True)
    taxonomy.write_text(
        json.dumps(
            [
                {
                    "label": "founder_execution",
                    "definition": "Evidence the founder can execute.",
                    "coarse_parent": "founding_team",
                }
            ]
        ),
        encoding="utf-8",
    )
    audit = inputs / "data/investors" / vc_slug / "audits" / f"{episode_slug}.json"
    audit.parent.mkdir(parents=True)
    audit.write_text(
        json.dumps(
            {
                "status": "audited",
                "vc_slug": vc_slug,
                "episode_slug": episode_slug,
                "pitch_sha256": sha256(pitch_bytes).hexdigest(),
                "leakage_checklist": {},
                "target_company_aliases": ["Example Company"],
            }
        ),
        encoding="utf-8",
    )
    manifest = (
        inputs / "data/investors" / vc_slug / "manifests" / f"{episode_slug}.json"
    )
    manifest.parent.mkdir(parents=True)
    files = [registry, pitch, audit, taxonomy, wiki / "principles.md"]
    manifest.write_text(
        json.dumps(
            {
                "vc_slug": vc_slug,
                "episode_slug": episode_slug,
                "files": [
                    {
                        "destination": path.relative_to(inputs).as_posix(),
                        "sha256": sha256(path.read_bytes()).hexdigest(),
                    }
                    for path in files
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return SimpleNamespace(
        wiki=wiki,
        pitch=pitch,
        registry=registry,
        taxonomy=taxonomy,
        audit=audit,
        manifest=manifest,
        precedents=None,
        precedent_manifest=None,
        portfolio_memory=None,
        target_company_aliases=("Example Company",),
    )


def write_config(
    root: Path,
    output: str = "outputs/fake-run",
    *,
    contract_version: str = "v1",
    precedents_enabled: bool = False,
) -> Path:
    path = root / f"run-{Path(output).name}.toml"
    path.write_text(
        '''[run]
vc_slug = "elizabeth-yin-hustle-fund"
episode_slug = "135-thoras-ai-the-twin-effect"
input_root = "inputs"
output_root = "{output}"
checkpoint_path = "{output}/checkpoints.sqlite"
contract_version = "{contract_version}"

[provider]
kind = "fake"
model = "deterministic-fixture"

[phase1]
min_iterations = 1
max_iterations = 2
max_precedent_searches = {max_precedent_searches}
max_precedent_reads = {max_precedent_reads}

[phase2]
min_iterations = 1
max_iterations = 2

[retrieval]
top_k = 3
max_exact_reads = 8

[precedents]
enabled = {precedents_enabled}
corpus_path = "data/investors/elizabeth-yin-hustle-fund/precedents"
allow_full_transcript = false
'''.format(
            output=output,
            contract_version=contract_version,
            precedents_enabled=str(precedents_enabled).lower(),
            max_precedent_searches=2 if precedents_enabled else 0,
            max_precedent_reads=4 if precedents_enabled else 0,
        ),
        encoding="utf-8",
    )
    return path


def write_v44_config(
    root: Path,
    output: str = "outputs/fake-v44",
    *,
    vc_slug: str = "elizabeth-yin-hustle-fund",
    episode_slug: str = "135-thoras-ai-the-twin-effect",
) -> Path:
    path = write_config(root, output=output, contract_version="v4.4")
    text = (
        path.read_text(encoding="utf-8")
        .replace('vc_slug = "elizabeth-yin-hustle-fund"', f'vc_slug = "{vc_slug}"')
        .replace(
            'episode_slug = "135-thoras-ai-the-twin-effect"',
            f'episode_slug = "{episode_slug}"',
        )
        .replace(
            'contract_version = "v4.4"',
            'contract_version = "v4.4"\nmode = "phase1_only"',
        )
    )
    text += '''

[embedding]
kind = "sentence_transformers"
model = "cli-v44-local"
revision = "pinned-revision"

[phase1_v44]
taxonomy_top_k = 5
claim_retrieval_top_k = 3
max_wiki_reads_per_claim = 2
max_precedent_reads_per_claim = 0
max_revisits = 1
'''
    path.write_text(text, encoding="utf-8")
    return path


def test_workflow_selects_isolated_v44_constructor(tmp_path: Path, monkeypatch) -> None:
    config = RunConfig.model_validate(
        {
            "run": {
                "vc_slug": "test-vc",
                "episode_slug": "18-rowvigor",
                "input_root": "inputs",
                "output_root": "outputs",
                "checkpoint_path": "outputs/checkpoints.sqlite",
                "contract_version": "v4.4",
                "mode": "phase1_only",
            },
            "provider": {"kind": "fake", "model": "fixture"},
            "phase1": {"min_iterations": 1, "max_iterations": 1},
            "phase2": {"min_iterations": 1, "max_iterations": 1},
            "retrieval": {"top_k": 3, "max_exact_reads": 8},
            "phase1_v44": {"max_precedent_reads_per_claim": 0},
        }
    )
    registry = tmp_path / "registry.toml"
    registry.write_text(
        'display_name = "Test Investor"\ncheck_tiers = ["small_exploratory"]\n',
        encoding="utf-8",
    )
    taxonomy = tmp_path / "taxonomy.json"
    taxonomy.write_text(
        json.dumps(
            [
                {
                    "label": "founder_execution",
                    "definition": "Founder execution evidence.",
                    "coarse_parent": "founding_team",
                }
            ]
        ),
        encoding="utf-8",
    )
    pitch = tmp_path / "pitch.txt"
    pitch.write_text("Founder reports five paid pilots.", encoding="utf-8")
    package = type(
        "Package",
        (),
        {"registry": registry, "taxonomy": taxonomy, "pitch": pitch},
    )()
    captured = {}

    class CapturingV44:
        def __init__(self, *args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs

    monkeypatch.setattr(cli_module, "_CliDecisionWorkflowV44", CapturingV44)
    sentinel_phase1 = object()
    sentinel_phase2 = object()
    result = cli_module._workflow(
        config,
        sentinel_phase1,
        sentinel_phase2,
        object(),
        package,
        object(),
    )

    assert isinstance(result, CapturingV44)
    assert captured["kwargs"]["phase1_provider"] is sentinel_phase1
    assert captured["kwargs"]["v44_settings"] == config.phase1_v44
    assert "phase2_provider" not in captured["kwargs"]


def test_cli_v44_preflight_run_and_verify_stay_phase1_only(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    package = _minimal_v44_package(tmp_path)
    config = write_v44_config(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli_module, "verify_package", lambda *_args, **_kwargs: package)
    monkeypatch.setattr(
        cli_module,
        "_embedding_provider",
        lambda *_args, **_kwargs: _LocalCliV44Embedder(),
    )

    assert main(["index", "--config", str(config)]) == 0
    capsys.readouterr()
    assert main(["preflight", "--config", str(config)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "ready"
    assert report["target_accessible"] is False
    assert report["phase2_model"] is None
    assert report["budgets"]["phase1_v44"] == {
        "taxonomy_top_k": 5,
        "claim_retrieval_top_k": 3,
        "max_wiki_reads_per_claim": 2,
        "max_precedent_reads_per_claim": 0,
        "max_revisits": 1,
    }

    assert main(["run", "--config", str(config)]) == 0
    summary = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert summary["phase1_status"] in {"accepted", "provisional"}
    assert summary["phase2_status"] == "not_run"
    assert main(["verify", "--config", str(config)]) == 0
    run_root = tmp_path / "outputs/fake-v44/135-thoras-ai-the-twin-effect"
    assert not (run_root / "phase2").exists()
    assert not list(run_root.glob("**/*phase2*response*.json"))


def test_cli_v44_fake_end_to_end_freezes_auditable_phase1_contract(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """Prove the public, offline CLI path without a network-backed dependency."""
    package = _minimal_v44_package(tmp_path)
    config = write_v44_config(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        cli_module,
        "_embedding_provider",
        lambda *_args, **_kwargs: _LocalCliV44Embedder(),
    )

    assert main(["index", "--config", str(config)]) == 0
    index_report = capsys.readouterr().out
    assert "indexed 1 wiki sections" in index_report
    semantic_index = json.loads(
        (
            tmp_path
            / "inputs/indexes/elizabeth-yin-hustle-fund.json"
        ).read_text(encoding="utf-8")
    )
    assert semantic_index["embedding_index"] == {
        "backend": "sentence_transformers",
        "coverage": 1.0,
        "dimension": 3,
        "document_prefix": "search_document: ",
        "embedded_count": 1,
        "model": "cli-v44-local",
        "normalize": True,
        "query_prefix": "search_query: ",
        "revision": "pinned-revision",
        "total_count": 1,
    }
    assert semantic_index["chunks"][0]["embedding"] is not None

    assert main(["preflight", "--config", str(config)]) == 0
    preflight = json.loads(capsys.readouterr().out)
    assert preflight["status"] == "ready"
    assert preflight["target_accessible"] is False

    assert main(["run", "--config", str(config)]) == 0
    summary = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert summary["phase1_status"] == "accepted"
    assert summary["phase2_status"] == "not_run"

    run_root = tmp_path / "outputs/fake-v44/135-thoras-ai-the-twin-effect"
    expected_frozen = {
        "phase1/claim-map.json",
        "phase1/claim-map.sha256",
        "phase1/claim-retrieval.json",
        "phase1/claim-retrieval.sha256",
        "phase1/taxonomy-neighborhood.json",
        "phase1/taxonomy-neighborhood.sha256",
        "phase1/adjudication.json",
        "phase1/adjudication.sha256",
        "phase1/investigation.json",
        "phase1/investigation.sha256",
        "state.json",
    }
    actual_frozen = {"state.json"}
    for digest_path in (run_root / "phase1").glob("*.sha256"):
        payload_path = digest_path.with_suffix(".json")
        assert payload_path.is_file()
        assert digest_path.read_text(encoding="ascii").strip() == sha256(
            payload_path.read_bytes()
        ).hexdigest()
        actual_frozen.add(payload_path.relative_to(run_root).as_posix())
        actual_frozen.add(digest_path.relative_to(run_root).as_posix())
    assert actual_frozen == expected_frozen

    claim_call = json.loads(
        (run_root / "phase1/claim-extraction/call-01.json").read_text(
            encoding="utf-8"
        )
    )
    adjudication_call = json.loads(
        (run_root / "phase1/adjudication/turn-01/call-01.json").read_text(
            encoding="utf-8"
        )
    )
    assert claim_call["phase"] == "phase1_claim_extraction"
    assert claim_call["parsed"]["schema_version"] == "claim-map-v4.4"
    assert adjudication_call["phase"] == "phase1_adjudication"
    assert (
        adjudication_call["parsed"]["schema_version"]
        == "rationale-adjudication-v4.4"
    )
    assert claim_call["provider_metadata"] == {
        "model": "deterministic-fixture",
        "provider": "fake",
    }
    assert adjudication_call["provider_metadata"] == claim_call[
        "provider_metadata"
    ]
    assert package.pitch.read_text(encoding="utf-8").strip() in claim_call["prompt"]
    combined_prompts = (
        claim_call["prompt"] + "\n" + adjudication_call["prompt"]
    ).casefold()
    assert "actual current target decision" not in combined_prompts
    assert "reference rationale" not in combined_prompts
    assert not list(run_root.glob("phase2/*model-response.json"))
    assert not list(run_root.glob("phase2/**/*model-response.json"))

    assert main(["verify", "--config", str(config)]) == 0


def test_cli_v44_run_verify_accepts_crlf_ml_labels_and_benign_sk_thread(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    vc_slug = "sk-learning-run-001"
    episode_slug = "crlf-label-training"
    pitch_bytes = (
        b"Founder reports five paid pilots.\r\n"
        b"Our service produces a gold label for each training item.\r\n"
        b"A ground truth label is retained for ML quality checks.\r\n"
    )
    _minimal_v44_package(
        tmp_path,
        vc_slug=vc_slug,
        episode_slug=episode_slug,
        pitch_bytes=pitch_bytes,
    )
    config = write_v44_config(
        tmp_path,
        vc_slug=vc_slug,
        episode_slug=episode_slug,
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        cli_module,
        "_embedding_provider",
        lambda *_args, **_kwargs: _LocalCliV44Embedder(),
    )

    assert main(["index", "--config", str(config)]) == 0
    capsys.readouterr()
    assert main(["run", "--config", str(config)]) == 0
    capsys.readouterr()
    assert main(["verify", "--config", str(config)]) == 0

    run_root = tmp_path / "outputs/fake-v44" / episode_slug
    owner = json.loads((run_root / "run-owner-v44.json").read_text(encoding="utf-8"))
    provenance = json.loads(
        (run_root / "input-provenance.json").read_text(encoding="utf-8")
    )
    assert owner["thread_id"] == f"{vc_slug}:{episode_slug}"
    assert "\r" not in owner["fingerprint_payload"]["pitch"]
    assert provenance["pitch_sha256"] == sha256(pitch_bytes).hexdigest()


def test_v44_verify_rejects_stale_package_provenance_and_run_config(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    package = _minimal_v44_package(tmp_path)
    config = write_v44_config(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        cli_module,
        "_embedding_provider",
        lambda *_args, **_kwargs: _LocalCliV44Embedder(),
    )
    assert main(["index", "--config", str(config)]) == 0
    capsys.readouterr()
    assert main(["run", "--config", str(config)]) == 0
    capsys.readouterr()
    run_root = tmp_path / "outputs/fake-v44/135-thoras-ai-the-twin-effect"
    verify_phase1_artifacts(run_root)

    provenance_path = run_root / "input-provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["package_manifest_sha256"] = "0" * 64
    provenance_path.write_bytes(canonical_bytes(provenance))
    with pytest.raises(ValueError, match="provenance|manifest|package"):
        verify_phase1_artifacts(run_root)

    provenance["package_manifest_sha256"] = sha256(
        package.manifest.read_bytes()
    ).hexdigest()
    provenance["package_manifest_path"] = str(package.pitch)
    provenance_path.write_bytes(canonical_bytes(provenance))
    with pytest.raises(ValueError, match="provenance|manifest|package|path"):
        verify_phase1_artifacts(run_root)

    provenance["package_manifest_path"] = str(package.manifest)
    provenance_path.write_bytes(canonical_bytes(provenance))
    run_config_path = run_root / "run-config.json"
    raw_config = json.loads(run_config_path.read_text(encoding="utf-8"))
    raw_config["phase1_v44"]["max_revisits"] = 0
    run_config_path.write_bytes(canonical_bytes(raw_config))
    with pytest.raises(ValueError, match="run-config|settings|fingerprint"):
        verify_phase1_artifacts(run_root)


@pytest.mark.parametrize(
    ("section", "field", "mutated"),
    [
        ("embedding", "batch_size", 17),
        ("embedding", "device", "cpu"),
        ("provider", "request_timeout_seconds", 301),
    ],
)
def test_v44_verify_rejects_mutated_full_execution_settings(
    tmp_path: Path,
    monkeypatch,
    capsys,
    section: str,
    field: str,
    mutated: object,
) -> None:
    _minimal_v44_package(tmp_path)
    config = write_v44_config(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        cli_module,
        "_embedding_provider",
        lambda *_args, **_kwargs: _LocalCliV44Embedder(),
    )
    assert main(["index", "--config", str(config)]) == 0
    capsys.readouterr()
    assert main(["run", "--config", str(config)]) == 0
    capsys.readouterr()
    run_root = tmp_path / "outputs/fake-v44/135-thoras-ai-the-twin-effect"
    run_config_path = run_root / "run-config.json"
    raw_config = json.loads(run_config_path.read_text(encoding="utf-8"))
    raw_config[section][field] = mutated
    run_config_path.write_bytes(canonical_bytes(raw_config))

    with pytest.raises(ValueError, match="run-config|settings|fingerprint"):
        verify_phase1_artifacts(run_root)


def test_cli_v44_audit_bootstrap_rejects_final_component_symlink(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    _minimal_v44_package(tmp_path)
    config = write_v44_config(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        cli_module,
        "_embedding_provider",
        lambda *_args, **_kwargs: _LocalCliV44Embedder(),
    )
    assert main(["index", "--config", str(config)]) == 0
    capsys.readouterr()
    external = tmp_path / "external-audit.json"
    external.write_text("sentinel\n", encoding="utf-8")
    run_root = tmp_path / "outputs/fake-v44/135-thoras-ai-the-twin-effect"
    run_root.mkdir(parents=True)
    (run_root / "run-config.json").symlink_to(external)

    assert main(["run", "--config", str(config)]) == 1
    assert external.read_text(encoding="utf-8") == "sentinel\n"


def test_cli_v44_audit_bootstrap_rejects_preexisting_mismatch(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    _minimal_v44_package(tmp_path)
    config = write_v44_config(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        cli_module,
        "_embedding_provider",
        lambda *_args, **_kwargs: _LocalCliV44Embedder(),
    )
    assert main(["index", "--config", str(config)]) == 0
    capsys.readouterr()
    run_root = tmp_path / "outputs/fake-v44/135-thoras-ai-the-twin-effect"
    run_root.mkdir(parents=True)
    audit_path = run_root / "run-config.json"
    original = b'{"tampered":true}\n'
    audit_path.write_bytes(original)

    assert main(["run", "--config", str(config)]) == 1
    assert audit_path.read_bytes() == original


def test_cli_v44_resume_requires_matching_immutable_audits(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    _minimal_v44_package(tmp_path)
    config = write_v44_config(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        cli_module,
        "_embedding_provider",
        lambda *_args, **_kwargs: _LocalCliV44Embedder(),
    )
    assert main(["index", "--config", str(config)]) == 0
    capsys.readouterr()
    assert main(["run", "--config", str(config)]) == 0
    capsys.readouterr()
    run_root = tmp_path / "outputs/fake-v44/135-thoras-ai-the-twin-effect"
    audit_names = (
        "run-config.json",
        "input-provenance.json",
        "wiki-sanitization.json",
    )
    before = {name: (run_root / name).read_bytes() for name in audit_names}

    assert main(["resume", "--config", str(config)]) == 0
    assert {name: (run_root / name).read_bytes() for name in audit_names} == before


@pytest.mark.parametrize(
    "audit_name",
    [
        "run-config.json",
        "input-provenance.json",
        "wiki-sanitization.json",
    ],
)
def test_cli_v44_verify_requires_public_audit_artifacts(
    tmp_path: Path, monkeypatch, capsys, audit_name: str
) -> None:
    _minimal_v44_package(tmp_path)
    config = write_v44_config(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        cli_module,
        "_embedding_provider",
        lambda *_args, **_kwargs: _LocalCliV44Embedder(),
    )
    assert main(["index", "--config", str(config)]) == 0
    capsys.readouterr()
    assert main(["run", "--config", str(config)]) == 0
    capsys.readouterr()
    run_root = tmp_path / "outputs/fake-v44/135-thoras-ai-the-twin-effect"
    (run_root / audit_name).unlink()

    assert main(["verify", "--config", str(config)]) == 1


def test_cli_v44_verify_binds_supplied_config_to_existing_run(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    _minimal_v44_package(tmp_path)
    config = write_v44_config(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        cli_module,
        "_embedding_provider",
        lambda *_args, **_kwargs: _LocalCliV44Embedder(),
    )
    assert main(["index", "--config", str(config)]) == 0
    capsys.readouterr()
    assert main(["run", "--config", str(config)]) == 0
    capsys.readouterr()
    changed = config.read_text(encoding="utf-8").replace(
        'model = "deterministic-fixture"',
        'model = "deterministic-fixture"\nrequest_timeout_seconds = 301',
        1,
    )
    config.write_text(changed, encoding="utf-8")

    assert main(["verify", "--config", str(config)]) == 1


def test_cli_v44_preflight_requires_taxonomy_document_embedding_readiness(
    tmp_path: Path, monkeypatch
) -> None:
    package = _minimal_v44_package(tmp_path)
    config = write_v44_config(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli_module, "verify_package", lambda *_args, **_kwargs: package)
    monkeypatch.setattr(
        cli_module,
        "_embedding_provider",
        lambda *_args, **_kwargs: _LocalCliV44Embedder(),
    )
    assert main(["index", "--config", str(config)]) == 0
    monkeypatch.setattr(
        cli_module,
        "_embedding_provider",
        lambda *_args, **_kwargs: _QueryOnlyCliV44Embedder(),
    )

    assert main(["preflight", "--config", str(config)]) == 1


def test_cli_rejects_decide_for_v44_before_any_provider_call(
    tmp_path: Path, monkeypatch
) -> None:
    config = write_v44_config(tmp_path)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("v4.4 decide must fail before provider construction")

    monkeypatch.setattr(cli_module, "_generation_provider", forbidden)
    assert main(
        [
            "decide",
            "--config",
            str(config),
            "--phase1-from",
            str(tmp_path / "phase1"),
        ]
    ) == 1


def add_precedent_corpus(
    root: Path, *, alias_bearing_slug: str | None = None
) -> Path:
    source = root / "precedent-source"
    source.mkdir()
    for slug in ("18-rowvigor", "135-thoras-ai-the-twin-effect", "150-after-target"):
        alias_text = (
            " about Thoras Ai The Twin Effect"
            if slug == alias_bearing_slug
            else ""
        )
        (source / f"{slug}.json").write_text(
            json.dumps(
                {
                    "transcript": (
                        f"Founder: {slug} pitch{alias_text}\nElizabeth: Thanks."
                    )
                }
            ),
            encoding="utf-8",
        )
    ledger = root / "precedent-ledger.json"
    ledger.write_text(
        json.dumps(
            [
                {
                    "episode_slug": slug,
                    "pitch_window_decision": "Unobserved",
                    "decision_context": "unclear",
                    "audit_notes": "No observed decision.",
                }
                for slug in ("18-rowvigor", "135-thoras-ai-the-twin-effect", "150-after-target")
            ]
        ),
        encoding="utf-8",
    )
    corpus = (
        root
        / "inputs/data/investors/elizabeth-yin-hustle-fund/precedents"
    )
    shutil.rmtree(corpus, ignore_errors=True)
    build_precedent_corpus(source, ledger, corpus, ("Elizabeth",))
    for manifest in (
        root / "inputs/data/investors/elizabeth-yin-hustle-fund/source-manifest.json",
        root
        / "inputs/data/investors/elizabeth-yin-hustle-fund/manifests/135-thoras-ai-the-twin-effect.json",
    ):
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        precedent_prefix = (
            "data/investors/elizabeth-yin-hustle-fund/precedents/"
        )
        payload["files"] = [
            row
            for row in payload["files"]
            if not row["destination"].startswith(precedent_prefix)
        ]
        payload["files"].extend(
            {
                "destination": path.relative_to(root / "inputs").as_posix(),
                "sha256": sha256(path.read_bytes()).hexdigest(),
            }
            for path in sorted(corpus.rglob("*"))
            if path.is_file()
        )
        payload["files"].sort(key=lambda row: row["destination"])
        manifest.write_text(json.dumps(payload), encoding="utf-8")
    return corpus


def test_cli_builds_separate_unfiltered_precedent_index(tmp_path: Path, monkeypatch) -> None:
    shutil.copytree(ROOT / "inputs", tmp_path / "inputs")
    add_precedent_corpus(tmp_path)
    config = write_config(tmp_path, contract_version="v3", precedents_enabled=True)
    monkeypatch.chdir(tmp_path)
    shutil.rmtree(tmp_path / "inputs/indexes", ignore_errors=True)

    assert main(["index", "--config", str(config)]) == 0

    path = tmp_path / "inputs/indexes/elizabeth-yin-hustle-fund.precedents.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["target_episode_slug"] is None
    assert {row["episode_slug"] for row in payload["episodes"]} == {
        "18-rowvigor",
        "135-thoras-ai-the-twin-effect",
        "150-after-target",
    }


def test_cli_preflight_missing_indexes_is_safe_and_never_constructs_provider(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    shutil.copytree(ROOT / "inputs", tmp_path / "inputs")
    add_precedent_corpus(tmp_path)
    config = write_config(tmp_path, contract_version="v3", precedents_enabled=True)
    monkeypatch.chdir(tmp_path)
    shutil.rmtree(tmp_path / "inputs/indexes", ignore_errors=True)

    def forbidden_provider(*args, **kwargs):
        raise AssertionError("preflight must not construct a provider")

    monkeypatch.setattr(cli_module, "_generation_provider", forbidden_provider)
    monkeypatch.setattr(cli_module, "_embedding_provider", forbidden_provider)

    assert main(["preflight", "--config", str(config)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "not_ready"
    assert report["missing_indexes"] == [
        "inputs/indexes/elizabeth-yin-hustle-fund.json",
        "inputs/indexes/elizabeth-yin-hustle-fund.precedents.json",
    ]
    assert report["index_command"] == f"uv run vc-clone-graph index --config {config}"
    assert report["target"] == "135-thoras-ai-the-twin-effect"
    assert report["pitch"]["status"] == "audited"
    assert len(report["pitch"]["sha256"]) == 64
    assert report["budgets"]["phase1"]["planning_max_output_tokens"] == 768
    assert report["budgets"]["phase2"]["planning_max_output_tokens"] == 768
    assert report["corpus_count"] == 3
    assert report["accessible_precedent_count"] == 2
    assert report["target_accessible"] is False


@pytest.mark.parametrize("embedding_kind", ["openai", "openrouter"])
def test_cli_rejects_remote_precedent_embeddings_before_provider_construction(
    tmp_path: Path, monkeypatch, embedding_kind: str
) -> None:
    config = write_config(
        tmp_path, contract_version="v3", precedents_enabled=True
    )
    config.write_text(
        config.read_text(encoding="utf-8")
        + f'''\n[embedding]\nkind = "{embedding_kind}"\nmodel = "remote-model"\n''',
        encoding="utf-8",
    )

    def forbidden_provider(*args, **kwargs):
        raise AssertionError("invalid config must not construct a provider")

    monkeypatch.setattr(cli_module, "_generation_provider", forbidden_provider)
    monkeypatch.setattr(cli_module, "_embedding_provider", forbidden_provider)
    monkeypatch.setattr(cli_module, "_indexing_provider", forbidden_provider)

    assert main(["index", "--config", str(config)]) == 1


def test_precedent_indexing_without_embedding_is_lexical_only(
    tmp_path: Path, monkeypatch
) -> None:
    config = RunConfig.model_validate(
        {
            "run": {
                "vc_slug": "elizabeth-yin-hustle-fund",
                "episode_slug": "135-thoras-ai-the-twin-effect",
                "input_root": "inputs",
                "output_root": "outputs",
                "checkpoint_path": "outputs/checkpoints.sqlite",
                "contract_version": "v3",
            },
            "provider": {"kind": "fake", "model": "fixture"},
            "phase1": {"min_iterations": 1, "max_iterations": 1},
            "phase2": {"min_iterations": 1, "max_iterations": 1},
            "retrieval": {"top_k": 3, "max_exact_reads": 8},
            "precedents": {"enabled": True},
        }
    )

    monkeypatch.setattr(
        cli_module,
        "_generation_provider",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("lexical-only indexing must not construct a provider")
        ),
    )

    assert cli_module._indexing_provider(config) is None


def test_cli_preflight_verifies_and_filters_indexes_without_generation(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    shutil.copytree(ROOT / "inputs", tmp_path / "inputs")
    add_precedent_corpus(tmp_path)
    config = write_config(tmp_path, contract_version="v3", precedents_enabled=True)
    monkeypatch.chdir(tmp_path)
    assert main(["index", "--config", str(config)]) == 0
    capsys.readouterr()

    def forbidden_provider(*args, **kwargs):
        raise AssertionError("preflight must not construct a provider")

    monkeypatch.setattr(cli_module, "_generation_provider", forbidden_provider)
    monkeypatch.setattr(cli_module, "_embedding_provider", forbidden_provider)

    assert main(["preflight", "--config", str(config)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "ready"
    assert report["corpus_count"] == 3
    assert report["accessible_precedent_count"] == 2
    assert report["target_accessible"] is False
    assert len(report["filtered_manifest_sha256"]) == 64
    assert report["provider"] == "fake"
    assert report["phase1_model"] == "deterministic-fixture"
    assert report["phase2_model"] == "deterministic-fixture"
    assert report["budgets"]["retrieval"] == {
        "max_exact_reads": 8,
        "top_k": 3,
        "precedent_selection_policy": "semantic",
        "precedent_candidate_pool_k": 30,
        "precedent_in_slots": 2,
        "precedent_out_slots": 2,
    }


def test_cli_preflight_run_and_decide_share_alias_aware_precedent_view(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    shutil.copytree(ROOT / "inputs", tmp_path / "inputs")
    add_precedent_corpus(tmp_path, alias_bearing_slug="18-rowvigor")
    config = write_config(tmp_path, precedents_enabled=True)
    replay_config = write_config(
        tmp_path,
        output="outputs/replay",
        precedents_enabled=True,
    )
    monkeypatch.chdir(tmp_path)
    assert main(["index", "--config", str(config)]) == 0
    capsys.readouterr()

    class NoGenerationProvider:
        generate_calls = 0

        def generate(self, request):
            self.generate_calls += 1
            raise AssertionError("parity test must not invoke provider generation")

    provider = NoGenerationProvider()
    monkeypatch.setattr(
        cli_module,
        "_generation_provider",
        lambda config, phase="phase1": provider,
    )

    captured: list[dict[str, object]] = []

    class CapturingWorkflow:
        def invoke(self, thread_id):
            return {}

        def invoke_phase2(self, thread_id, investigation, digest, **kwargs):
            return {}

    def capture_workflow(*args, **kwargs):
        captured.append(
            {
                "corpus": kwargs["precedent_corpus"],
                "manifest_sha256": kwargs["precedent_manifest_sha256"],
            }
        )
        return CapturingWorkflow()

    monkeypatch.setattr(cli_module, "_workflow", capture_workflow)

    assert main(["preflight", "--config", str(config)]) == 0
    preflight = json.loads(capsys.readouterr().out)
    assert main(["run", "--config", str(config)]) == 0

    runtime_corpus = captured[0]["corpus"]
    runtime_manifest = runtime_corpus.filtered_manifest()
    runtime_slugs = {row.episode_slug for row in runtime_corpus.list_episodes()}
    assert runtime_slugs == {"150-after-target"}

    frozen_root = tmp_path / "frozen-source"
    frozen_phase1 = frozen_root / "phase1"
    frozen_phase1.mkdir(parents=True)
    wiki_payload = json.loads(
        (
            tmp_path
            / "inputs/indexes/elizabeth-yin-hustle-fund.json"
        ).read_text(encoding="utf-8")
    )
    investigation = cli_module.Investigation.model_validate(
        phase1(wiki_payload["chunks"][0]["chunk_id"])
    )
    raw_investigation = canonical_bytes(investigation)
    (frozen_phase1 / "investigation.json").write_bytes(raw_investigation)
    (frozen_phase1 / "investigation.sha256").write_text(
        sha256(raw_investigation).hexdigest() + "\n", encoding="ascii"
    )
    (frozen_root / "precedent-manifest.filtered.json").write_text(
        json.dumps(runtime_manifest.model_dump(mode="json")), encoding="utf-8"
    )
    (frozen_root / "state.json").write_text(
        json.dumps(
            {
                "accessible_precedent_count": len(runtime_slugs),
                "precedent_manifest_sha256": runtime_manifest.sha256,
            }
        ),
        encoding="utf-8",
    )

    assert main(
        [
            "decide",
            "--config",
            str(replay_config),
            "--phase1-from",
            str(frozen_phase1),
        ]
    ) == 0

    replay_corpus = captured[1]["corpus"]
    replay_manifest = replay_corpus.filtered_manifest()
    assert preflight["accessible_precedent_count"] == len(runtime_slugs)
    assert preflight["filtered_manifest_sha256"] == runtime_manifest.sha256
    assert replay_corpus.list_episodes() == runtime_corpus.list_episodes()
    assert replay_manifest == runtime_manifest
    assert all(
        row["manifest_sha256"] == runtime_manifest.sha256 for row in captured
    )
    assert provider.generate_calls == 0


def test_cli_writes_filtered_manifest_before_first_generation_call(
    tmp_path: Path, monkeypatch
) -> None:
    shutil.copytree(ROOT / "inputs", tmp_path / "inputs")
    add_precedent_corpus(tmp_path)
    config = write_config(tmp_path, contract_version="v3", precedents_enabled=True)
    monkeypatch.chdir(tmp_path)
    assert main(["index", "--config", str(config)]) == 0
    run_root = tmp_path / "outputs/fake-run/135-thoras-ai-the-twin-effect"

    class ManifestAssertingProvider:
        def __init__(self):
            self.generate_calls = 0

        def embed(self, texts):
            return [[float(len(text)), 1.0] for text in texts]

        def generate(self, request):
            self.generate_calls += 1
            manifest_path = run_root / "precedent-manifest.filtered.json"
            assert manifest_path.is_file()
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            assert manifest["target_episode_slug"] == "135-thoras-ai-the-twin-effect"
            assert all(
                row["episode_slug"] != manifest["target_episode_slug"]
                for row in manifest["accessible_episodes"]
            )
            unsigned = {key: value for key, value in manifest.items() if key != "sha256"}
            assert manifest["sha256"] == sha256(
                json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            raise RuntimeError("stop after manifest timing assertion")

    provider = ManifestAssertingProvider()
    monkeypatch.setattr(cli_module, "_generation_provider", lambda config, phase="phase1": provider)

    assert main(["run", "--config", str(config)]) == 1
    assert provider.generate_calls == 1


def test_cli_rejects_tampered_precedent_index_before_generation(
    tmp_path: Path, monkeypatch
) -> None:
    shutil.copytree(ROOT / "inputs", tmp_path / "inputs")
    add_precedent_corpus(tmp_path)
    config = write_config(tmp_path, contract_version="v3", precedents_enabled=True)
    monkeypatch.chdir(tmp_path)
    assert main(["index", "--config", str(config)]) == 0
    index_path = tmp_path / "inputs/indexes/elizabeth-yin-hustle-fund.precedents.json"
    tampered = json.loads(index_path.read_text(encoding="utf-8"))
    tampered["episodes"][0]["investor_aliases"] = ["Tampered"]
    index_path.write_text(json.dumps(tampered), encoding="utf-8")

    class NoGenerationProvider:
        generate_calls = 0

        def embed(self, texts):
            return [[float(len(text)), 1.0] for text in texts]

        def generate(self, request):
            self.generate_calls += 1
            raise AssertionError("generation must not start")

    provider = NoGenerationProvider()
    monkeypatch.setattr(cli_module, "_generation_provider", lambda config, phase="phase1": provider)

    assert main(["run", "--config", str(config)]) == 1
    assert provider.generate_calls == 0


def test_cli_requires_verified_corpus_when_precedents_enabled(
    tmp_path: Path, monkeypatch
) -> None:
    shutil.copytree(ROOT / "inputs", tmp_path / "inputs")
    shutil.rmtree(
        tmp_path / "inputs/data/investors/elizabeth-yin-hustle-fund/precedents"
    )
    config = write_config(tmp_path, contract_version="v3", precedents_enabled=True)
    monkeypatch.chdir(tmp_path)
    provider_started = False

    def forbidden_provider(config):
        nonlocal provider_started
        provider_started = True
        raise AssertionError("indexing provider must not be constructed")

    monkeypatch.setattr(cli_module, "_indexing_provider", forbidden_provider)

    assert main(["index", "--config", str(config)]) == 1
    assert provider_started is False


@pytest.mark.parametrize("failure", ["missing-index", "corpus", "source"])
def test_cli_precedent_integrity_failures_stop_before_generation(
    tmp_path: Path, monkeypatch, failure: str
) -> None:
    shutil.copytree(ROOT / "inputs", tmp_path / "inputs")
    corpus = add_precedent_corpus(tmp_path)
    config = write_config(tmp_path, contract_version="v3", precedents_enabled=True)
    monkeypatch.chdir(tmp_path)
    assert main(["index", "--config", str(config)]) == 0
    if failure == "missing-index":
        (
            tmp_path
            / "inputs/indexes/elizabeth-yin-hustle-fund.precedents.json"
        ).unlink()
    elif failure == "corpus":
        path = corpus / "corpus-manifest.json"
        path.write_bytes(path.read_bytes() + b" ")
    else:
        path = corpus / "sources/18-rowvigor.json"
        path.write_bytes(path.read_bytes() + b" ")

    class NoGenerationProvider:
        generate_calls = 0

        def embed(self, texts):
            return [[float(len(text)), 1.0] for text in texts]

        def generate(self, request):
            self.generate_calls += 1
            raise AssertionError("generation must not start")

    provider = NoGenerationProvider()
    monkeypatch.setattr(
        cli_module,
        "_generation_provider",
        lambda config, phase="phase1": provider,
    )

    assert main(["run", "--config", str(config)]) == 1
    assert provider.generate_calls == 0


def test_cli_indexes_runs_and_verifies_fake_workflow(tmp_path: Path, monkeypatch, capsys) -> None:
    shutil.copytree(ROOT / "inputs", tmp_path / "inputs")
    config = write_config(tmp_path)
    monkeypatch.chdir(tmp_path)

    assert main(["index", "--config", str(config)]) == 0
    assert main(["run", "--config", str(config)]) == 0
    run_summary = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert run_summary == {
        "any_check_decision": "In",
        "any_check_likelihood": 0.65,
        "decision": "In",
        "investment_likelihood": 0.65,
        "phase1_status": "accepted",
        "phase1_iterations": 1,
        "phase1_findings": [],
        "phase2_status": "accepted",
        "phase2_iterations": 1,
        "phase2_findings": [],
        "ranking_score": 0.75,
        "recommended_check_tier": "small_exploratory",
        "standard_check_decision": "Out",
        "standard_check_likelihood": 0.3,
    }
    assert main(["verify", "--config", str(config)]) == 0
    summary = tmp_path / "outputs/fake-run/135-thoras-ai-the-twin-effect/summary.json"
    assert summary.is_file()
    sanitization = tmp_path / "outputs/fake-run/135-thoras-ai-the-twin-effect/wiki-sanitization.json"
    report = json.loads(sanitization.read_text())
    assert report["alias_hashes"]
    assert "Thoras" not in sanitization.read_text()


def test_cli_runs_and_verifies_v2_contract(tmp_path: Path, monkeypatch) -> None:
    shutil.copytree(ROOT / "inputs", tmp_path / "inputs")
    config = write_config(
        tmp_path, output="outputs/fake-v2", contract_version="v2"
    )
    monkeypatch.chdir(tmp_path)

    assert main(["index", "--config", str(config)]) == 0
    assert main(["run", "--config", str(config)]) == 0
    assert main(["verify", "--config", str(config)]) == 0
    run_root = tmp_path / "outputs/fake-v2/135-thoras-ai-the-twin-effect"
    investigation = json.loads((run_root / "phase1/investigation.json").read_text())
    decision = json.loads((run_root / "phase2/decision.json").read_text())
    assert investigation["schema_version"] == "investigation-v2"
    assert decision["schema_version"] == "decision-v2"


def test_cli_verify_rejects_modified_frozen_decision(tmp_path: Path, monkeypatch) -> None:
    shutil.copytree(ROOT / "inputs", tmp_path / "inputs")
    config = write_config(tmp_path)
    monkeypatch.chdir(tmp_path)
    assert main(["index", "--config", str(config)]) == 0
    assert main(["run", "--config", str(config)]) == 0
    decision = tmp_path / "outputs/fake-run/135-thoras-ai-the-twin-effect/phase2/decision.json"
    decision.write_text(decision.read_text() + " ")
    assert main(["verify", "--config", str(config)]) == 1


def test_cli_propagates_sanitization_quality_to_phase1_status(
    tmp_path: Path, monkeypatch
) -> None:
    shutil.copytree(ROOT / "inputs", tmp_path / "inputs")
    config = write_config(tmp_path)
    monkeypatch.chdir(tmp_path)
    assert main(["index", "--config", str(config)]) == 0
    original = HybridWikiIndex.for_pitch

    def degraded(self, aliases):
        index, report = original(self, aliases)
        return index, replace(
            report, quality_findings=("SANITIZED_EMBEDDING_FALLBACK",)
        )

    monkeypatch.setattr(HybridWikiIndex, "for_pitch", degraded)
    assert main(["run", "--config", str(config)]) == 0
    state = json.loads(
        (
            tmp_path
            / "outputs/fake-run/135-thoras-ai-the-twin-effect/state.json"
        ).read_text()
    )
    assert state["phase1_status"] == "provisional"
    assert "SANITIZED_EMBEDDING_FALLBACK" in state["phase1_findings"]


def test_cli_resume_rewrites_state_artifact(tmp_path: Path, monkeypatch) -> None:
    shutil.copytree(ROOT / "inputs", tmp_path / "inputs")
    config = write_config(tmp_path)
    monkeypatch.chdir(tmp_path)
    assert main(["index", "--config", str(config)]) == 0
    assert main(["run", "--config", str(config)]) == 0
    state = tmp_path / "outputs/fake-run/135-thoras-ai-the-twin-effect/state.json"
    state.unlink()

    assert main(["resume", "--config", str(config)]) == 0
    assert state.is_file()


def test_cli_decide_replays_only_phase2_from_frozen_phase1(tmp_path: Path, monkeypatch, capsys) -> None:
    shutil.copytree(ROOT / "inputs", tmp_path / "inputs")
    source_config = write_config(tmp_path)
    replay_config = write_config(tmp_path, "outputs/replay")
    monkeypatch.chdir(tmp_path)
    assert main(["index", "--config", str(source_config)]) == 0
    assert main(["run", "--config", str(source_config)]) == 0
    source_phase1 = tmp_path / "outputs/fake-run/135-thoras-ai-the-twin-effect/phase1"

    assert main([
        "decide", "--config", str(replay_config), "--phase1-from", str(source_phase1)
    ]) == 0
    replay_summary = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert replay_summary["any_check_decision"] == "In"
    assert replay_summary["standard_check_decision"] == "Out"
    assert replay_summary["recommended_check_tier"] == "small_exploratory"
    replay_root = tmp_path / "outputs/replay/135-thoras-ai-the-twin-effect"
    assert (replay_root / "phase1/investigation.json").is_file()
    assert (replay_root / "phase1-provenance.json").is_file()
    assert (replay_root / "phase2/decision.json").is_file()
    trace = (replay_root / "phase2/turn-01/decision-model-response.json").read_text()
    assert "phase1_plan" not in trace


def test_cli_v4_decide_replays_only_phase2_from_frozen_phase1(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    slug = "135-thoras-ai-the-twin-effect"

    def bound_investigation(request):
        payload = investigation_v4(request)
        payload["episode_slug"] = slug
        return payload

    def bound_decision(request):
        payload = decision_v4(request)
        payload["episode_slug"] = slug
        return payload

    shutil.copytree(ROOT / "inputs", tmp_path / "inputs")
    source_config = write_config(
        tmp_path, output="outputs/v4-source", contract_version="v4"
    )
    replay_config = write_config(
        tmp_path, output="outputs/v4-replay", contract_version="v4"
    )
    monkeypatch.chdir(tmp_path)
    assert main(["index", "--config", str(source_config)]) == 0

    source_provider = ScriptedProviderV4(
        [plan_v4(), bound_investigation, plan_v4(), bound_decision]
    )
    monkeypatch.setattr(
        cli_module,
        "_generation_provider",
        lambda config, phase="phase1": source_provider,
    )
    assert main(["run", "--config", str(source_config)]) == 0
    source_phase1 = tmp_path / f"outputs/v4-source/{slug}/phase1"

    replay_provider = ScriptedProviderV4([plan_v4(), bound_decision])
    monkeypatch.setattr(
        cli_module,
        "_generation_provider",
        lambda config, phase="phase1": replay_provider,
    )
    assert main(
        [
            "decide",
            "--config",
            str(replay_config),
            "--phase1-from",
            str(source_phase1),
        ]
    ) == 0
    assert main(["verify", "--config", str(replay_config)]) == 0

    assert [request.phase for request in replay_provider.requests] == [
        "phase2_plan",
        "phase2",
    ]
    replay_root = tmp_path / f"outputs/v4-replay/{slug}"
    assert (replay_root / "phase1/investigation.json").is_file()
    assert (replay_root / "phase2/decision.json").is_file()
    summary = json.loads(capsys.readouterr().out.strip().splitlines()[-2])
    assert summary["decision"] == "In"


def test_cli_v3_replay_carries_phase1_audit_and_verifies(
    tmp_path: Path, monkeypatch
) -> None:
    class AuditedFakeProvider(FakeProvider):
        def generate(self, request):
            result = super().generate(request)
            return result.model_copy(
                update={
                    "raw_metadata": {
                        "provider": "fake",
                        "model": "scripted-v3",
                    }
                }
            )

    shutil.copytree(ROOT / "inputs", tmp_path / "inputs")
    add_precedent_corpus(tmp_path)
    source_config = write_config(
        tmp_path,
        output="outputs/v3-source",
        contract_version="v3",
        precedents_enabled=True,
    )
    replay_config = write_config(
        tmp_path,
        output="outputs/v3-replay",
        contract_version="v3",
        precedents_enabled=True,
    )
    monkeypatch.chdir(tmp_path)
    assert main(["index", "--config", str(source_config)]) == 0
    index = HybridWikiIndex.load(
        tmp_path / "inputs/indexes/elizabeth-yin-hustle-fund.json",
        tmp_path / "inputs/wiki/elizabeth-yin-hustle-fund",
        None,
    )
    chunk_id = index.search("founder", 1)[0].chunk_id
    taxonomy_labels = {
        row["label"]
        for row in json.loads(
            (tmp_path / "inputs/taxonomy/codebook_v_final.json").read_text(
                encoding="utf-8"
            )
        )
    }
    investigation_payload = phase1_v3(
        chunk_id,
        [],
        saturated=True,
        taxonomy_labels=taxonomy_labels,
    )
    investigation_payload["rationales"][0]["pitch_evidence"] = [
        "five design partnerships"
    ]
    investigation = InvestigationV3.model_validate(investigation_payload)
    digest = sha256(canonical_bytes(investigation)).hexdigest()
    decision_payload = as_decision_v3(decision(phase1(chunk_id)))
    decision_payload["investigation_sha256"] = digest
    phase1_plan = {
        "questions": ["Founder pattern?", "Exception pattern?"],
        "wiki_queries": ["founder"],
        "precedent_queries": [],
        "transcript_reads": [],
        "decision_reads": [],
        "reconsideration_focus": "Test the execution pattern.",
    }
    source_provider = AuditedFakeProvider(
        outputs=[
            phase1_plan,
            investigation_payload,
            phase1_plan,
            investigation_payload,
            phase2_plan(),
            decision_payload,
        ]
    )
    monkeypatch.setattr(
        cli_module,
        "_generation_provider",
        lambda config, phase="phase1": source_provider,
    )

    assert main(["run", "--config", str(source_config)]) == 0
    source_root = tmp_path / "outputs/v3-source/135-thoras-ai-the-twin-effect"
    source_state_path = source_root / "state.json"
    source_state = json.loads(source_state_path.read_text())
    for name in ("missing-state", "history", "usage"):
        bad_config = write_config(
            tmp_path,
            output=f"outputs/v3-replay-{name}",
            contract_version="v3",
            precedents_enabled=True,
        )
        if name == "missing-state":
            source_state_path.unlink()
        else:
            tampered = json.loads(json.dumps(source_state))
            if name == "history":
                tampered["query_history"].append("tampered query")
            else:
                tampered["usage_by_phase"]["phase1"]["input_tokens"] += 1
            source_state_path.write_text(json.dumps(tampered), encoding="utf-8")
        forbidden = AuditedFakeProvider(outputs=[])
        monkeypatch.setattr(
            cli_module,
            "_generation_provider",
            lambda config, phase="phase1", provider=forbidden: provider,
        )
        assert main(
            [
                "decide",
                "--config",
                str(bad_config),
                "--phase1-from",
                str(source_root / "phase1"),
            ]
        ) == 1
        assert forbidden.requests == []
        source_state_path.write_text(json.dumps(source_state), encoding="utf-8")
    replay_provider = AuditedFakeProvider(outputs=[phase2_plan(), decision_payload])
    monkeypatch.setattr(
        cli_module,
        "_generation_provider",
        lambda config, phase="phase1": replay_provider,
    )

    assert main(
        [
            "decide",
            "--config",
            str(replay_config),
            "--phase1-from",
            str(source_root / "phase1"),
        ]
    ) == 0
    assert main(["verify", "--config", str(replay_config)]) == 0
    replay_root = tmp_path / "outputs/v3-replay/135-thoras-ai-the-twin-effect"
    replay_state = json.loads((replay_root / "state.json").read_text())
    assert replay_state["phase1_iteration"] == source_state["phase1_iteration"]
    assert replay_state["query_history"] == source_state["query_history"]
    assert replay_state["phase1_precedent_reads"] == source_state[
        "phase1_precedent_reads"
    ]
    assert replay_state["usage_by_phase"]["phase1"] == source_state[
        "usage_by_phase"
    ]["phase1"]
    assert [request.phase for request in replay_provider.requests] == [
        "phase2_plan",
        "phase2",
    ]


def test_cli_decide_rejects_output_nested_inside_source(tmp_path: Path, monkeypatch) -> None:
    shutil.copytree(ROOT / "inputs", tmp_path / "inputs")
    source_config = write_config(tmp_path)
    monkeypatch.chdir(tmp_path)
    assert main(["index", "--config", str(source_config)]) == 0
    assert main(["run", "--config", str(source_config)]) == 0
    source_phase1 = tmp_path / "outputs/fake-run/135-thoras-ai-the-twin-effect/phase1"
    nested = source_phase1 / "nested-output"
    nested_config = write_config(tmp_path, str(nested))

    assert main([
        "decide", "--config", str(nested_config), "--phase1-from", str(source_phase1)
    ]) == 1


def test_cli_decide_rejects_existing_checkpoint(tmp_path: Path, monkeypatch) -> None:
    shutil.copytree(ROOT / "inputs", tmp_path / "inputs")
    source_config = write_config(tmp_path)
    replay_config = write_config(tmp_path, "outputs/replay")
    replay_config.write_text(
        replay_config.read_text().replace(
            'checkpoint_path = "outputs/replay/checkpoints.sqlite"',
            'checkpoint_path = "outputs/fake-run/checkpoints.sqlite"',
        )
    )
    monkeypatch.chdir(tmp_path)
    assert main(["index", "--config", str(source_config)]) == 0
    assert main(["run", "--config", str(source_config)]) == 0
    source_phase1 = tmp_path / "outputs/fake-run/135-thoras-ai-the-twin-effect/phase1"

    assert main([
        "decide", "--config", str(replay_config), "--phase1-from", str(source_phase1)
    ]) == 1


def test_cli_decide_rejects_checkpoint_inside_frozen_source(tmp_path: Path, monkeypatch) -> None:
    shutil.copytree(ROOT / "inputs", tmp_path / "inputs")
    source_config = write_config(tmp_path)
    replay_config = write_config(tmp_path, "outputs/replay")
    monkeypatch.chdir(tmp_path)
    assert main(["index", "--config", str(source_config)]) == 0
    assert main(["run", "--config", str(source_config)]) == 0
    source_phase1 = tmp_path / "outputs/fake-run/135-thoras-ai-the-twin-effect/phase1"
    unsafe_checkpoint = source_phase1 / "replay.sqlite"
    unsafe_checkpoint_relative = Path(
        "outputs/fake-run/135-thoras-ai-the-twin-effect/phase1/replay.sqlite"
    )
    replay_config.write_text(
        replay_config.read_text().replace(
            'checkpoint_path = "outputs/replay/checkpoints.sqlite"',
            f'checkpoint_path = "{unsafe_checkpoint_relative}"',
        )
    )

    assert main([
        "decide", "--config", str(replay_config), "--phase1-from", str(source_phase1)
    ]) == 1
    assert not unsafe_checkpoint.exists()


def test_cli_builds_openrouter_generation_with_local_ollama_embedding(
    monkeypatch,
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    config = RunConfig.model_validate(
        {
            "run": {
                "vc_slug": "elizabeth-yin-hustle-fund",
                "episode_slug": "135-thoras-ai-the-twin-effect",
                "input_root": "inputs",
                "output_root": "outputs/openrouter",
                "checkpoint_path": "outputs/openrouter/checkpoints.sqlite",
            },
            "provider": {
                "kind": "openrouter",
                "model": "openai/gpt-5.6-luna",
                "base_url": "https://openrouter.ai/api/v1",
                "api_key_env": "OPENROUTER_API_KEY",
            },
            "embedding": {
                "kind": "ollama",
                "model": "nomic-embed-text",
                "base_url": "http://127.0.0.1:11434",
            },
            "phase1": {"min_iterations": 1, "max_iterations": 2},
            "phase2": {"min_iterations": 1, "max_iterations": 2},
            "retrieval": {"top_k": 5, "max_exact_reads": 16},
        }
    )

    provider = _provider(config)

    assert isinstance(provider, CompositeProvider)
    assert isinstance(provider.generator, OpenRouterProvider)
    assert isinstance(provider.embedder, OllamaProvider)


def test_cli_builds_distinct_phase_specific_openrouter_generators(
    monkeypatch,
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    config = RunConfig.model_validate(
        {
            "run": {
                "vc_slug": "charles-hudson-precursor-ventures",
                "episode_slug": "20-harper-wilde",
                "input_root": "inputs",
                "output_root": "outputs/openrouter",
                "checkpoint_path": "outputs/openrouter/checkpoints.sqlite",
            },
            "provider": {
                "kind": "openrouter",
                "model": "default/model",
                "api_key_env": "OPENROUTER_API_KEY",
                "max_output_tokens": 4096,
            },
            "phase1": {
                "min_iterations": 1,
                "max_iterations": 2,
                "model": "openai/gpt-5.6-luna-phase1",
                "max_output_tokens": 2048,
            },
            "phase2": {
                "min_iterations": 1,
                "max_iterations": 2,
                "model": "openai/gpt-5.6-luna-phase2",
                "max_output_tokens": 8192,
            },
            "retrieval": {"top_k": 5, "max_exact_reads": 16},
        }
    )

    phase1 = _generation_provider(config, "phase1")
    phase2 = _generation_provider(config, "phase2")

    assert isinstance(phase1, OpenRouterProvider)
    assert isinstance(phase2, OpenRouterProvider)
    assert phase1 is not phase2
    assert phase1.model == "openai/gpt-5.6-luna-phase1"
    assert phase2.model == "openai/gpt-5.6-luna-phase2"
    assert phase1.max_output_tokens == 2048
    assert phase2.max_output_tokens == 8192


def test_cli_builds_ollama_generation_with_independent_openai_embedding(
    monkeypatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    config = RunConfig.model_validate(
        {
            "run": {
                "vc_slug": "charles-hudson-precursor-ventures",
                "episode_slug": "20-harper-wilde",
                "input_root": "inputs",
                "output_root": "outputs/ollama",
                "checkpoint_path": "outputs/ollama/checkpoints.sqlite",
            },
            "provider": {
                "kind": "ollama",
                "model": "qwen3.5:9b",
                "base_url": "http://127.0.0.1:11434",
            },
            "embedding": {
                "kind": "openai",
                "model": "text-embedding-3-small",
            },
            "phase1": {"min_iterations": 1, "max_iterations": 2},
            "phase2": {"min_iterations": 1, "max_iterations": 2},
            "retrieval": {"top_k": 5, "max_exact_reads": 16},
        }
    )

    provider = _provider(config)

    assert isinstance(provider, CompositeProvider)
    assert isinstance(provider.generator, OllamaProvider)
    assert provider.generator.embedding_model is None
    assert isinstance(provider.embedder, OpenAIProvider)
    assert provider.embedder.embedding_model == "text-embedding-3-small"
