from __future__ import annotations

from pathlib import Path

import pytest

from vc_clone_graph.personalized_config import load_personalized_config


def write_valid_config(tmp_path: Path, replacements: dict[str, str] | None = None) -> Path:
    text = '''[data]
source_registry = "evaluation/canonical_runs_v4_v41_portfolio_2026-08-15.json"
input_root = "inputs"
rationale_references = "evaluation/phase1_ground_truth_rationales"
taxonomy = "../agentic-vc-clone-framework/taxonomy/codebook_v_final.json"

[embedding]
model = "nomic-ai/nomic-embed-text-v1.5"
revision = "e9b6763023c676ca8431644204f50c2b100d9aab"
device = "auto"
batch_size = 16
normalize = true
document_prefix = "search_document: "
query_prefix = "search_query: "

[evaluation]
outer_group = "episode_slug"
seed = 20260821
inner_splits = 3
classification_metric = "macro_balanced_accuracy"
ranking_metric = "macro_within_vc_average_precision"
review_budgets = [5, 10, 20]

[tabpfn]
enabled = true
n_estimators = [4, 8]
pca_components = [16, 32]
device = "auto"

[setfit]
enabled = true
model = "sentence-transformers/all-MiniLM-L6-v2"
revision = "c9745ed1d9f207416be6d2e6f8de32d1f16199bf"
device = "auto"
lambda_rank = [0.25, 0.5]
lambda_aux = [0.1, 0.25]
epochs = [1, 2]
max_pairs = 4096

[graph]
enabled = false
run_only_after_primary = true
hidden_dimensions = 32
layers = 2
dropout = 0.5
weight_decay = 0.0001
epochs = 50

[ensemble]
enabled = true
minimum_disagreement_rate = 0.10
require_inner_improvement = true

[execution]
output = "reports/evaluation/multimodal-personalized-2026-08-21"
cache = "outputs/cache/multimodal-personalized-v1"
max_retries = 1
api_cost_ceiling_usd = 0.0
'''
    for old, new in (replacements or {}).items():
        text = text.replace(old, new)
    path = tmp_path / "experiment.toml"
    path.write_text(text, encoding="utf-8")
    return path


def test_config_locks_episode_grouping_and_model_bounds(tmp_path: Path) -> None:
    config = load_personalized_config(write_valid_config(tmp_path))

    assert config.evaluation.outer_group == "episode_slug"
    assert config.evaluation.review_budgets == (5, 10, 20)
    assert config.tabpfn.n_estimators == (4, 8)
    assert config.tabpfn.pca_components == (16, 32)
    assert config.setfit.lambda_rank == (0.25, 0.5)
    assert config.setfit.device == "auto"
    assert config.graph.weight_decay == 0.0001
    assert config.graph.epochs == 50
    assert config.execution.max_retries == 1
    assert config.execution.api_cost_ceiling_usd == 0.0


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ('outer_group = "episode_slug"', 'outer_group = "row"', "episode_slug"),
        (
            'revision = "e9b6763023c676ca8431644204f50c2b100d9aab"',
            'revision = ""',
            "revision",
        ),
        ("api_cost_ceiling_usd = 0.0", "api_cost_ceiling_usd = -1.0", "cost"),
        ("n_estimators = [4, 8]", "n_estimators = [0, 8]", "n_estimators"),
        ("weight_decay = 0.0001", "weight_decay = -0.1", "weight_decay"),
    ],
)
def test_config_rejects_unsafe_or_unbounded_values(
    tmp_path: Path, old: str, new: str, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        load_personalized_config(write_valid_config(tmp_path, {old: new}))


def test_config_rejects_unknown_keys(tmp_path: Path) -> None:
    path = write_valid_config(
        tmp_path,
        {"max_retries = 1": "max_retries = 1\nunknown_switch = true"},
    )
    with pytest.raises(ValueError, match="unknown"):
        load_personalized_config(path)
