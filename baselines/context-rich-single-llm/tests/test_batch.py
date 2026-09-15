import json

from config import load_baseline_config
from batch import execute_context, pilot_slugs
from prompt import BaselineContext, PrecedentContext, WikiDocument
from vc_clone_graph.providers.base import GenerationResult, Usage

from test_schema_prompt import payload


class CountingProvider:
    def __init__(self, parsed):
        self.parsed = parsed
        self.calls = []

    def generate(self, request):
        self.calls.append(request)
        content = json.dumps(self.parsed)
        return GenerationResult(parsed=self.parsed, content=content, usage=Usage(input_tokens=100, output_tokens=50, cost_usd=.01), elapsed_seconds=.2, raw_metadata={"physical_attempts": 1})


def context():
    slugs = ("18-example", "2-p", "3-p", "4-p", "5-p")
    return BaselineContext("39-example", "Charles Hudson", "PITCH", (WikiDocument("persona.md", "WIKI"),), ({"label": "founder_execution"},), tuple(PrecedentContext(slug, "TRANSCRIPT", "Out", "NO", .8) for slug in slugs))


def test_execute_calls_provider_once_and_preserves_usage_and_label(tmp_path):
    provider = CountingProvider(payload())
    summary = execute_context(context(), {"source": "test"}, provider, tmp_path, "In", "high", 16384)
    assert len(provider.calls) == 1
    assert summary["status"] == "completed" and summary["actual_label"] == "In"
    assert summary["usage"]["cost_usd"] == .01
    assert json.loads((tmp_path / "provider-response.json").read_text())["content"]
    assert "actual_label" not in (tmp_path / "prompt.txt").read_text()


def test_schema_failure_is_recorded_without_second_call(tmp_path):
    provider = CountingProvider({"bad": True})
    summary = execute_context(context(), {}, provider, tmp_path, "Out", "high", 16384)
    assert len(provider.calls) == 1 and summary["status"] == "failed"
    assert not (tmp_path / "result.json").exists()
    assert (tmp_path / "validation-findings.json").exists()


def test_unselected_descriptive_precedent_fails_but_selected_prefix_is_valid(tmp_path):
    good = payload()
    good["decision"]["decisive_precedents"] = ["18-example — observed Out"]
    assert execute_context(context(), {}, CountingProvider(good), tmp_path / "good", "In", "high", 16384)["status"] == "completed"
    bad = payload()
    bad["decision"]["decisive_precedents"] = ["999-unknown — observed In"]
    summary = execute_context(context(), {}, CountingProvider(bad), tmp_path / "bad", "In", "high", 16384)
    assert summary["status"] == "failed"
    assert not (tmp_path / "bad" / "result.json").exists()


def test_selected_precedent_allows_trailing_citation_punctuation(tmp_path):
    value = payload()
    value["decision"]["decisive_precedents"] = [
        "18-example: observed Out",
        "2-p, observed Out",
        "3-p; observed Out",
    ]

    summary = execute_context(
        context(), {}, CountingProvider(value), tmp_path, "In", "high", 16384
    )

    assert summary["status"] == "completed"


def test_pilot_selects_first_in_and_out_from_round_robin():
    assert pilot_slugs((("18-out", "Out"), ("39-in", "In"), ("41-in", "In"))) == ("39-in", "18-out")
