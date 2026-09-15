import json
from pathlib import Path

import pytest

from vc_clone_graph.rehearsal_config import RehearsalConfig
from vc_clone_graph.rehearsal_runtime import list_investors, resolve_investor
from vc_clone_graph.retrieval import HybridWikiIndex


ROOT = Path(__file__).resolve().parents[1]


class TinyEmbedder:
    metadata = {
        "backend": "sentence_transformers",
        "model": "tiny",
        "revision": "revision",
        "normalize": True,
        "document_prefix": "search_document: ",
        "query_prefix": "search_query: ",
    }

    def embed(self, texts):
        return self.embed_documents(texts)

    def embed_documents(self, texts):
        return [[float(len(text)), 1.0] for text in texts]

    def embed_queries(self, texts):
        return [[float(len(text)), 1.0] for text in texts]


class NoCallProvider:
    model = "no-call"

    def generate(self, request):  # pragma: no cover - runtime composition only
        raise AssertionError("generation is not expected")


def _fixture(tmp_path: Path) -> tuple[RehearsalConfig, str]:
    inputs = tmp_path / "inputs"
    vc_slug = "test-investor"
    wiki = inputs / "wiki" / vc_slug
    wiki.mkdir(parents=True)
    (wiki / "principles.md").write_text(
        "# Conflict\nTargetCo is an existing portfolio company.\n\n"
        "# Founder\nExecution and customer learning matter.\n",
        encoding="utf-8",
    )
    registry = inputs / "investors" / f"{vc_slug}.toml"
    registry.parent.mkdir(parents=True)
    registry.write_text(
        f'vc_slug = "{vc_slug}"\n'
        'display_name = "Test Investor"\n'
        'firm = "Test Fund"\n'
        'role = "Partner"\n'
        f'wiki_path = "wiki/{vc_slug}"\n',
        encoding="utf-8",
    )
    taxonomy = inputs / "taxonomy/codebook_v_final.json"
    taxonomy.parent.mkdir(parents=True)
    taxonomy.write_text(
        json.dumps(
            [
                {
                    "label": "founder_execution",
                    "definition": "Ability to execute.",
                    "coarse_parent": "founding_team",
                }
            ]
        ),
        encoding="utf-8",
    )
    index = HybridWikiIndex.build(wiki, TinyEmbedder(), require_complete_embeddings=True)
    index_path = inputs / "indexes" / f"{vc_slug}.json"
    index.save(index_path)
    config = RehearsalConfig.model_validate(
        {
            "rehearsal": {
                "input_root": "inputs",
                "output_root": "outputs/rehearsals",
                "checkpoint_path": "outputs/rehearsals/checkpoints.sqlite",
            },
            "provider": {"kind": "fake", "model": "fixture"},
            "embedding": {
                "kind": "sentence_transformers",
                "model": "tiny",
                "revision": "revision",
            },
            "retrieval": {"top_k": 3, "max_exact_reads": 5},
            "precedents": {"enabled": False},
            "portfolio_memory": {"enabled": False},
        }
    )
    return config, vc_slug


def test_all_six_investors_are_discovered() -> None:
    investors = list_investors(ROOT / "inputs")

    assert {row.vc_slug for row in investors} == {
        "charles-hudson-precursor-ventures",
        "cyan-banister-long-journey-ventures",
        "elizabeth-yin-hustle-fund",
        "jesse-middleton-flybridge",
        "jillian-manus-structure-capital",
        "phil-nadel",
    }


def test_runtime_filters_target_company_from_wiki(tmp_path: Path, monkeypatch) -> None:
    config, vc_slug = _fixture(tmp_path)
    monkeypatch.chdir(tmp_path)

    runtime = resolve_investor(
        config,
        vc_slug,
        target_company_aliases=("TargetCo",),
        provider=NoCallProvider(),
        embedder=TinyEmbedder(),
    )
    hits = runtime.wiki.search("portfolio conflict", limit=20)

    assert all("TargetCo" not in hit.excerpt for hit in hits)


def test_rehearsal_runtime_does_not_require_episode_slug(tmp_path: Path, monkeypatch) -> None:
    config, vc_slug = _fixture(tmp_path)
    monkeypatch.chdir(tmp_path)

    runtime = resolve_investor(
        config,
        vc_slug,
        provider=NoCallProvider(),
        embedder=TinyEmbedder(),
    )

    assert runtime.investor.vc_slug == vc_slug
    assert runtime.precedents is None
    assert runtime.portfolio is None
    assert runtime.taxonomy[0]["label"] == "founder_execution"
    assert runtime.classifier_resolution.status == "fallback"
    assert "registry" in (runtime.classifier_resolution.reason or "")


def test_runtime_can_require_classifier_instead_of_falling_back(
    tmp_path: Path, monkeypatch
) -> None:
    config, vc_slug = _fixture(tmp_path)
    payload = config.model_dump(mode="python")
    payload["classification"]["fallback_to_rationale_only"] = False
    strict = RehearsalConfig.model_validate(payload)
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValueError, match="classifier required"):
        resolve_investor(
            strict,
            vc_slug,
            provider=NoCallProvider(),
            embedder=TinyEmbedder(),
        )


def test_grounded_runtime_resolves_exact_historical_baseline(
    tmp_path: Path, monkeypatch
) -> None:
    config, vc_slug = _fixture(tmp_path)
    payload = config.model_dump(mode="python")
    payload["classification"]["mode"] = "v41_grounded"
    grounded = RehearsalConfig.model_validate(payload)
    expected = object()
    calls = []

    def fake_load(**kwargs):
        calls.append(kwargs)
        return expected

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "vc_clone_graph.rehearsal_runtime.load_canonical_baseline", fake_load
    )
    runtime = resolve_investor(
        grounded,
        vc_slug,
        excluded_episode_slug="18-example",
        provider=NoCallProvider(),
        embedder=TinyEmbedder(),
    )

    assert runtime.grounded_baseline is expected
    assert calls[0]["canonical_vc_slug"] == "test-investor"
    assert calls[0]["episode_slug"] == "18-example"


def test_grounded_runtime_accepts_a_verified_live_baseline(
    tmp_path: Path, monkeypatch
) -> None:
    config, vc_slug = _fixture(tmp_path)
    payload = config.model_dump(mode="python")
    payload["classification"]["mode"] = "v41_grounded"
    grounded = RehearsalConfig.model_validate(payload)
    expected = object()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "vc_clone_graph.rehearsal_runtime.load_canonical_baseline",
        lambda **kwargs: pytest.fail("registry loader must not be called"),
    )

    runtime = resolve_investor(
        grounded,
        vc_slug,
        excluded_episode_slug="live-session-one",
        grounded_baseline=expected,
        provider=NoCallProvider(),
        embedder=TinyEmbedder(),
    )

    assert runtime.grounded_baseline is expected


def test_missing_grounded_registry_fails_before_generation(
    tmp_path: Path, monkeypatch
) -> None:
    config, vc_slug = _fixture(tmp_path)
    payload = config.model_dump(mode="python")
    payload["classification"]["mode"] = "v41_grounded"
    grounded = RehearsalConfig.model_validate(payload)
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValueError, match="canonical baseline"):
        resolve_investor(
            grounded,
            vc_slug,
            excluded_episode_slug="18-example",
            provider=NoCallProvider(),
            embedder=TinyEmbedder(),
        )
