from pathlib import Path

import pytest

from vc_clone_graph.config import load_config


def write_config(
    tmp_path: Path,
    *,
    output_root: str = "outputs",
    run_extra: str = "",
    extra: str = "",
) -> Path:
    path = tmp_path / "run.toml"
    path.write_text(
        f'''[run]
vc_slug = "elizabeth-yin-hustle-fund"
episode_slug = "135-thoras-ai-the-twin-effect"
input_root = "inputs"
output_root = "{output_root}"
checkpoint_path = "outputs/checkpoints.sqlite"
{run_extra}

[provider]
kind = "ollama"
model = "qwen3.5:9b"
base_url = "http://127.0.0.1:11434"
embedding_model = "qwen3-embedding:0.6b"

[phase1]
min_iterations = 1
max_iterations = 2

[phase2]
min_iterations = 1
max_iterations = 2

[retrieval]
top_k = 6
max_exact_reads = 12
{extra}
''',
        encoding="utf-8",
    )
    return path


def test_loads_provider_independent_run_config(tmp_path: Path) -> None:
    config = load_config(write_config(tmp_path))
    assert config.provider.kind == "ollama"
    assert config.provider.model == "qwen3.5:9b"
    assert config.phase1.max_iterations == 2
    assert config.run.taxonomy_path == "taxonomy/codebook_v_final.json"
    assert config.run.contract_version == "v1"
    assert config.phase1.model is None
    assert config.phase1.max_precedent_searches == 0
    assert config.precedents.enabled is False
    assert config.precedents.selection_policy == "semantic"
    assert config.precedents.candidate_pool_k == 30
    assert config.precedents.in_slots == 2
    assert config.precedents.out_slots == 2


def test_loads_versioned_taxonomy_path(tmp_path: Path) -> None:
    config = load_config(
        write_config(
            tmp_path,
            run_extra='taxonomy_path = "taxonomy/codebook_v2.json"',
        )
    )

    assert config.run.taxonomy_path == "taxonomy/codebook_v2.json"


def test_loads_v2_contract_version(tmp_path: Path) -> None:
    config = load_config(
        write_config(tmp_path, run_extra='contract_version = "v2"')
    )

    assert config.run.contract_version == "v2"


def test_loads_v4_contract_version(tmp_path: Path) -> None:
    config = load_config(
        write_config(tmp_path, run_extra='contract_version = "v4"')
    )

    assert config.run.contract_version == "v4"


def test_loads_v41_contract_version_without_changing_v4(tmp_path: Path) -> None:
    config = load_config(
        write_config(tmp_path, run_extra='contract_version = "v4.1"')
    )

    assert config.run.contract_version == "v4.1"


@pytest.mark.parametrize("version", ["v4", "v4.1"])
def test_loads_canonical_v4_phase1_only_experiment(
    tmp_path: Path, version: str
) -> None:
    config = load_config(write_config(
        tmp_path,
        run_extra=f'contract_version = "{version}"\nmode = "phase1_only"',
    ))
    assert config.run.contract_version == version
    assert config.run.mode == "phase1_only"


def test_loads_v5_full_contract_with_portfolio_memory(tmp_path: Path) -> None:
    config = load_config(
        write_config(
            tmp_path,
            run_extra='contract_version = "v5"',
            extra='''
[embedding]
kind = "sentence_transformers"
model = "nomic-ai/nomic-embed-text-v1.5"
revision = "main"

[portfolio_memory]
enabled = true
retrieval_top_k = 4
candidate_pool_k = 20
''',
        )
    )

    assert config.run.contract_version == "v5"
    assert config.portfolio_memory.enabled is True


def test_loads_v42_phase1_only_contract(tmp_path: Path) -> None:
    config = load_config(
        write_config(
            tmp_path,
            run_extra='contract_version = "v4.2"\nmode = "phase1_only"',
        )
    )

    assert config.run.contract_version == "v4.2"
    assert config.run.mode == "phase1_only"


def test_loads_v43_phase1_only_contract_with_portfolio_memory(tmp_path: Path) -> None:
    path = write_config(
        tmp_path,
        run_extra='contract_version = "v4.3"\nmode = "phase1_only"',
        extra='''

[embedding]
kind = "sentence_transformers"
model = "nomic-ai/nomic-embed-text-v1.5"
revision = "e9b6763023c676ca8431644204f50c2b100d9aab"

[portfolio_memory]
enabled = true
''',
    )
    config = load_config(path)
    assert config.run.contract_version == "v4.3"
    assert config.run.mode == "phase1_only"
    assert config.portfolio_memory.enabled is True


def test_loads_v44_phase1_only_contract_settings(tmp_path: Path) -> None:
    config = load_config(
        write_config(
            tmp_path,
            run_extra='contract_version = "v4.4"\nmode = "phase1_only"',
            extra='''

[phase1_v44]
taxonomy_top_k = 5
claim_retrieval_top_k = 3
max_wiki_reads_per_claim = 2
max_precedent_reads_per_claim = 2
max_revisits = 1
''',
        )
    )

    assert config.run.contract_version == "v4.4"
    assert config.run.mode == "phase1_only"
    assert config.phase1_v44 is not None
    assert config.phase1_v44.taxonomy_top_k == 5
    assert config.phase1_v44.claim_retrieval_top_k == 3
    assert config.phase1_v44.max_wiki_reads_per_claim == 2
    assert config.phase1_v44.max_precedent_reads_per_claim == 2
    assert config.phase1_v44.max_revisits == 1


def test_v44_rejects_full_execution(tmp_path: Path) -> None:
    path = write_config(
        tmp_path,
        run_extra='contract_version = "v4.4"',
        extra='''

[phase1_v44]
''',
    )

    with pytest.raises(ValueError, match="v4.4 requires phase1_only"):
        load_config(path)


def test_v44_rejects_missing_phase1_settings(tmp_path: Path) -> None:
    path = write_config(
        tmp_path,
        run_extra='contract_version = "v4.4"\nmode = "phase1_only"',
    )

    with pytest.raises(ValueError, match="v4.4 requires phase1_v44"):
        load_config(path)


def test_phase1_v44_settings_reject_non_v44_contract(tmp_path: Path) -> None:
    path = write_config(
        tmp_path,
        extra='''

[phase1_v44]
''',
    )

    with pytest.raises(ValueError, match="phase1_v44 requires contract_version=v4.4"):
        load_config(path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("taxonomy_top_k", 0),
        ("taxonomy_top_k", 13),
        ("claim_retrieval_top_k", 0),
        ("claim_retrieval_top_k", 11),
        ("max_wiki_reads_per_claim", 0),
        ("max_wiki_reads_per_claim", 7),
        ("max_precedent_reads_per_claim", -1),
        ("max_precedent_reads_per_claim", 7),
        ("max_revisits", -1),
        ("max_revisits", 2),
    ],
)
def test_v44_rejects_phase1_setting_outside_bounds(
    tmp_path: Path, field: str, value: int
) -> None:
    path = write_config(
        tmp_path,
        run_extra='contract_version = "v4.4"\nmode = "phase1_only"',
        extra=f'''\n[phase1_v44]\n{field} = {value}\n''',
    )

    with pytest.raises(ValueError):
        load_config(path)


def test_v44_accepts_portfolio_memory(tmp_path: Path) -> None:
    path = write_config(
        tmp_path,
        run_extra='contract_version = "v4.4"\nmode = "phase1_only"',
        extra='''

[embedding]
kind = "sentence_transformers"
model = "nomic-ai/nomic-embed-text-v1.5"
revision = "abc123"

[portfolio_memory]
enabled = true

[phase1_v44]
''',
    )

    assert load_config(path).portfolio_memory.enabled is True


def test_loads_v3_phase_controls_and_precedent_settings(tmp_path: Path) -> None:
    path = write_config(
        tmp_path,
        run_extra='contract_version = "v3"',
        extra='''

[precedents]
enabled = true
corpus_path = "data/investors/charles-hudson-precursor-ventures/precedents"
allow_full_transcript = true
selection_policy = "contrastive"
candidate_pool_k = 24
in_slots = 3
out_slots = 3
''',
    )
    contents = path.read_text(encoding="utf-8")
    contents = contents.replace(
        '''kind = "ollama"
model = "qwen3.5:9b"
base_url = "http://127.0.0.1:11434"
embedding_model = "qwen3-embedding:0.6b"''',
        '''kind = "openrouter"
model = "openai/gpt-5.6-luna"''',
    )
    contents = contents.replace(
        "[phase1]\nmin_iterations = 1\nmax_iterations = 2",
        '''[phase1]
min_iterations = 1
max_iterations = 2
model = "openai/gpt-5.6-luna"
max_output_tokens = 2048
reasoning_effort = "high"
planning_max_output_tokens = 1536
planning_reasoning_effort = "low"
max_precedent_searches = 3
max_precedent_reads = 8''',
    )
    path.write_text(contents, encoding="utf-8")

    config = load_config(path)

    assert config.run.contract_version == "v3"
    assert config.phase1.model == "openai/gpt-5.6-luna"
    assert config.phase1.max_output_tokens == 2048
    assert config.phase1.reasoning_effort == "high"
    assert config.phase1.planning_max_output_tokens == 1536
    assert config.phase1.planning_reasoning_effort == "low"
    assert config.phase1.max_precedent_searches == 3
    assert config.phase1.max_precedent_reads == 8
    assert config.precedents.enabled is True
    assert config.precedents.allow_full_transcript is True
    assert config.precedents.selection_policy == "contrastive"
    assert config.precedents.candidate_pool_k == 24
    assert config.precedents.in_slots == 3
    assert config.precedents.out_slots == 3


@pytest.mark.parametrize(
    "settings",
    [
        'selection_policy = "invalid"',
        "candidate_pool_k = 3\nin_slots = 2\nout_slots = 2",
        "candidate_pool_k = 30\nin_slots = 0\nout_slots = 0",
    ],
)
def test_rejects_invalid_contrastive_precedent_settings(
    tmp_path: Path, settings: str
) -> None:
    path = write_config(
        tmp_path,
        extra=f'''\n[precedents]\nenabled = true\n{settings}\n''',
    )

    with pytest.raises(ValueError):
        load_config(path)


@pytest.mark.parametrize("embedding_kind", ["openai", "openrouter"])
def test_precedents_reject_remote_embedding_provider_before_construction(
    tmp_path: Path, embedding_kind: str
) -> None:
    path = write_config(
        tmp_path,
        extra=f'''\n
[embedding]
kind = "{embedding_kind}"
model = "remote-embedding-model"

[precedents]
enabled = true
''',
    )

    with pytest.raises(ValueError, match="precedent embeddings must be local"):
        load_config(path)


@pytest.mark.parametrize(
    "embedding_block",
    [
        "",
        '''
[embedding]
kind = "ollama"
model = "nomic-embed-text"
base_url = "http://127.0.0.1:11434"
''',
    ],
)
def test_precedents_accept_lexical_only_or_local_ollama_embeddings(
    tmp_path: Path, embedding_block: str
) -> None:
    config = load_config(
        write_config(
            tmp_path,
            extra=f'''{embedding_block}
[precedents]
enabled = true
''',
        )
    )

    assert config.embedding is None or config.embedding.kind == "ollama"


def test_wiki_only_run_retains_remote_embedding_flexibility(tmp_path: Path) -> None:
    config = load_config(
        write_config(
            tmp_path,
            extra='''

[embedding]
kind = "openai"
model = "text-embedding-3-small"
''',
        )
    )

    assert config.precedents.enabled is False
    assert config.embedding.kind == "openai"


@pytest.mark.parametrize("provider_kind", ["ollama", "openai"])
@pytest.mark.parametrize("phase", ["phase1", "phase2"])
def test_rejects_reasoning_effort_for_non_openrouter_provider(
    tmp_path: Path, provider_kind: str, phase: str
) -> None:
    path = write_config(tmp_path)
    contents = path.read_text(encoding="utf-8").replace(
        'kind = "ollama"', f'kind = "{provider_kind}"'
    )
    marker = f"[{phase}]\nmin_iterations = 1\nmax_iterations = 2"
    contents = contents.replace(marker, f'{marker}\nreasoning_effort = "medium"')
    path.write_text(contents, encoding="utf-8")

    with pytest.raises(ValueError, match="only supported.*openrouter"):
        load_config(path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_output_tokens", 255),
        ("max_output_tokens", 16385),
        ("reasoning_effort", '"extreme"'),
        ("max_precedent_searches", 21),
        ("max_precedent_reads", 101),
    ],
)
def test_rejects_invalid_phase_control_bounds(
    tmp_path: Path, field: str, value: object
) -> None:
    path = write_config(tmp_path)
    rendered = value if isinstance(value, str) else str(value)
    path.write_text(
        path.read_text(encoding="utf-8").replace(
            "max_iterations = 2", f"max_iterations = 2\n{field} = {rendered}", 1
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        load_config(path)


@pytest.mark.parametrize("value", ["../precedents", "/tmp/precedents", "a\\b"])
def test_rejects_unsafe_precedent_corpus_path(tmp_path: Path, value: str) -> None:
    with pytest.raises(ValueError, match="safe relative"):
        load_config(
            write_config(
                tmp_path,
                extra=f'''\n[precedents]\ncorpus_path = "{value}"\n''',
            )
        )


@pytest.mark.parametrize("value", ["../taxonomy.json", "/tmp/taxonomy.json", "a\\b"])
def test_rejects_unsafe_taxonomy_path(tmp_path: Path, value: str) -> None:
    with pytest.raises(ValueError, match="safe relative"):
        load_config(
            write_config(tmp_path, run_extra=f'taxonomy_path = "{value}"')
        )


@pytest.mark.parametrize("value", ["/tmp/out", "../out", "a\\b"])
def test_rejects_unsafe_output_path(tmp_path: Path, value: str) -> None:
    with pytest.raises(ValueError, match="safe relative"):
        load_config(write_config(tmp_path, output_root=value))


def test_rejects_unknown_configuration_keys(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Extra inputs"):
        load_config(write_config(tmp_path, extra="surprise = 1"))


def test_rejects_iteration_minimum_above_maximum(tmp_path: Path) -> None:
    path = write_config(tmp_path)
    path.write_text(
        path.read_text(encoding="utf-8").replace(
            "min_iterations = 1\nmax_iterations = 2",
            "min_iterations = 3\nmax_iterations = 2",
            1,
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="minimum"):
        load_config(path)


def test_loads_openrouter_generation_with_independent_ollama_embedding(
    tmp_path: Path,
) -> None:
    path = tmp_path / "openrouter.toml"
    path.write_text(
        '''[run]
vc_slug = "elizabeth-yin-hustle-fund"
episode_slug = "135-thoras-ai-the-twin-effect"
input_root = "inputs"
output_root = "outputs/openrouter"
checkpoint_path = "outputs/openrouter/checkpoints.sqlite"

[provider]
kind = "openrouter"
model = "openai/gpt-5.6-luna"
base_url = "https://openrouter.ai/api/v1"
api_key_env = "OPENROUTER_API_KEY"
require_parameters = true
data_collection = "deny"

[embedding]
kind = "ollama"
model = "nomic-embed-text"
base_url = "http://127.0.0.1:11434"

[phase1]
min_iterations = 1
max_iterations = 2

[phase2]
min_iterations = 1
max_iterations = 2

[retrieval]
top_k = 5
max_exact_reads = 16
''',
        encoding="utf-8",
    )

    config = load_config(path)

    assert config.provider.kind == "openrouter"
    assert config.provider.api_key_env == "OPENROUTER_API_KEY"
    assert config.provider.require_parameters is True
    assert config.provider.data_collection == "deny"
    assert config.embedding is not None
    assert config.embedding.kind == "ollama"


def test_loads_in_process_sentence_transformer_embedding_configuration(
    tmp_path: Path,
) -> None:
    path = write_config(tmp_path)
    path.write_text(
        path.read_text(encoding="utf-8")
        + '''
[embedding]
kind = "sentence_transformers"
model = "nomic-ai/nomic-embed-text-v1.5"
revision = "abc123"
device = "cuda"
batch_size = 16
normalize = true
document_prefix = "search_document: "
query_prefix = "search_query: "
require_complete_index = true
''',
        encoding="utf-8",
    )

    config = load_config(path)

    assert config.embedding is not None
    assert config.embedding.kind == "sentence_transformers"
    assert config.embedding.model == "nomic-ai/nomic-embed-text-v1.5"
    assert config.embedding.revision == "abc123"
    assert config.embedding.device == "cuda"
    assert config.embedding.batch_size == 16
    assert config.embedding.normalize is True
    assert config.embedding.require_complete_index is True


def test_charles_full_precedent_canary_has_locked_luna_budgets() -> None:
    config = load_config(
        Path(__file__).resolve().parents[1]
        / "configs/openrouter-luna-charles-full-precedent-canary.toml"
    )

    assert config.run.episode_slug == "20-harper-wilde"
    assert config.run.input_root == "inputs"
    assert config.run.contract_version == "v3"
    assert config.provider.kind == "openrouter"
    assert config.provider.model == "openai/gpt-5.6-luna"
    assert config.provider.api_key_env == "OPENROUTER_API_KEY"
    assert config.provider.data_collection == "deny"
    assert config.provider.require_parameters is True
    assert config.provider.max_output_tokens == 16384
    assert config.embedding is not None
    assert config.embedding.kind == "ollama"
    assert config.embedding.model == "nomic-embed-text"
    assert config.embedding.base_url == "http://127.0.0.1:11434"
    for phase in (config.phase1, config.phase2):
        assert phase.model == "openai/gpt-5.6-luna"
        assert phase.reasoning_effort == "high"
        assert (phase.min_iterations, phase.max_iterations) == (2, 3)
        assert phase.max_output_tokens == 16384
        assert phase.planning_max_output_tokens == 2048
        assert phase.planning_reasoning_effort == "low"
        assert phase.max_precedent_searches == 4
        assert phase.max_precedent_reads == 12
    assert config.retrieval.top_k == 6
    assert config.retrieval.max_exact_reads == 30
    assert config.precedents.enabled is True
    assert config.precedents.corpus_path == (
        "data/investors/charles-hudson-precursor-ventures/precedents"
    )
    assert config.precedents.allow_full_transcript is True
    assert config.run.output_root != config.run.checkpoint_path
    assert config.embedding.model == "nomic-embed-text"


def test_charles_v4_planners_have_extended_repair_budget() -> None:
    config = load_config(
        Path(__file__).resolve().parents[1]
        / "configs/openrouter-luna-charles-v4.toml"
    )

    assert config.run.contract_version == "v4"
    assert config.phase1.planning_max_output_tokens == 8192
    assert config.phase2.planning_max_output_tokens == 8192


def test_loads_strict_portfolio_memory_configuration(tmp_path: Path) -> None:
    path = write_config(
        tmp_path,
        run_extra='contract_version = "v4.1"',
        extra='''
[embedding]
kind = "sentence_transformers"
model = "nomic-ai/nomic-embed-text-v1.5"
revision = "abc123"

[portfolio_memory]
enabled = true
corpus_path = "data/investors/<vc_slug>/portfolio-memory"
retrieval_top_k = 5
candidate_pool_k = 15
require_complete_embeddings = true
''',
    )
    config = load_config(path)
    assert config.portfolio_memory.enabled is True
    assert config.portfolio_memory.retrieval_top_k == 5


def test_v42_accepts_temporal_portfolio_memory(tmp_path: Path) -> None:
    path = write_config(
        tmp_path,
        run_extra='contract_version = "v4.2"\nmode = "phase1_only"',
        extra='''
[embedding]
kind = "sentence_transformers"
model = "nomic-ai/nomic-embed-text-v1.5"
revision = "abc123"

[portfolio_memory]
enabled = true
corpus_path = "data/investors/<vc_slug>/portfolio-memory"
''',
    )
    assert load_config(path).portfolio_memory.enabled is True


def test_rejects_unsafe_portfolio_memory_path(tmp_path: Path) -> None:
    path = write_config(
        tmp_path,
        run_extra='contract_version = "v4.1"',
        extra='''
[embedding]
kind = "sentence_transformers"
model = "nomic-ai/nomic-embed-text-v1.5"
revision = "abc123"

[portfolio_memory]
enabled = true
corpus_path = "../portfolio-memory"
''',
    )
    with pytest.raises(ValueError, match="path"):
        load_config(path)
