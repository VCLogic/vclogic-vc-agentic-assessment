import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from openai import APIConnectionError, APIResponseValidationError, APIStatusError, APITimeoutError

from vc_clone_graph.providers.base import (
    CompositeProvider,
    GenerationRequest,
    GenerationResult,
    Usage,
)
from vc_clone_graph.providers.fake import FakeProvider
from vc_clone_graph.providers.ollama import OllamaProvider
from vc_clone_graph.providers.openai import OpenAIProvider
from vc_clone_graph.providers.openrouter import OpenRouterProvider
from vc_clone_graph.providers.sentence_transformers import (
    SentenceTransformerEmbeddingProvider,
)


SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
    "additionalProperties": False,
}


def test_fake_provider_returns_scripted_structured_output_and_usage() -> None:
    provider = FakeProvider(outputs=[{"answer": "yes"}])
    result = provider.generate(GenerationRequest("test", "prompt", SCHEMA))
    assert result.parsed == {"answer": "yes"}
    assert result.usage.input_tokens > 0
    assert provider.requests[0].phase == "test"


def test_ollama_provider_sends_schema_and_records_native_usage() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert request.url.path == "/api/chat"
        assert body["format"] == SCHEMA
        assert body["stream"] is False
        assert body["think"] is False
        assert body["options"]["num_predict"] == 4096
        return httpx.Response(
            200,
            json={
                "message": {"role": "assistant", "content": '{"answer":"yes"}'},
                "prompt_eval_count": 12,
                "eval_count": 4,
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://test")
    provider = OllamaProvider("qwen", "embed", "http://test", client=client)
    result = provider.generate(GenerationRequest("test", "prompt", SCHEMA))
    assert result.parsed == {"answer": "yes"}
    assert result.usage.input_tokens == 12
    assert result.usage.output_tokens == 4
    assert result.elapsed_seconds >= 0


def test_ollama_provider_can_use_json_mode_with_graph_validation() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["format"] == "json"
        assert body["options"]["num_ctx"] == 8192
        assert '"additionalProperties":false' in body["messages"][0]["content"]
        return httpx.Response(200, json={"message": {"content": '{"answer":"yes"}'}})

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://test")
    provider = OllamaProvider(
        "qwen",
        "embed",
        "http://test",
        client=client,
        structured_output_mode="json",
        context_window=8192,
    )

    assert provider.generate(GenerationRequest("test", "prompt", SCHEMA)).parsed == {
        "answer": "yes"
    }


def test_ollama_embedding_adapter_returns_vectors() -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/embed"
        text = json.loads(request.content)["input"][0]
        seen.append(text)
        vector = [1.0, 0.0] if text == "one" else [0.0, 1.0]
        return httpx.Response(200, json={"embeddings": [vector]})

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://test")
    provider = OllamaProvider("qwen", "embed", "http://test", client=client)
    assert provider.embed(["one", "two"]) == [[1.0, 0.0], [0.0, 1.0]]
    assert seen == ["one", "two"]


def test_sentence_transformer_uses_distinct_document_and_query_tasks() -> None:
    class RecordingModel:
        def __init__(self) -> None:
            self.calls: list[tuple[list[str], dict]] = []

        def encode(self, texts, **kwargs):
            self.calls.append((list(texts), kwargs))
            return [[float(index), 1.0] for index, _ in enumerate(texts)]

    model = RecordingModel()
    provider = SentenceTransformerEmbeddingProvider(
        "nomic-ai/nomic-embed-text-v1.5",
        revision="abc123",
        device="cpu",
        batch_size=8,
        normalize=True,
        document_prefix="search_document: ",
        query_prefix="search_query: ",
        model_instance=model,
    )

    documents = provider.embed_documents(["founder transcript", "market evidence"])
    queries = provider.embed_queries(["founder-market fit"])

    assert documents == [[0.0, 1.0], [1.0, 1.0]]
    assert queries == [[0.0, 1.0]]
    assert model.calls[0][0] == [
        "search_document: founder transcript",
        "search_document: market evidence",
    ]
    assert model.calls[1][0] == ["search_query: founder-market fit"]
    assert model.calls[0][1]["batch_size"] == 8
    assert model.calls[0][1]["normalize_embeddings"] is True
    assert provider.metadata["model"] == "nomic-ai/nomic-embed-text-v1.5"
    assert provider.metadata["revision"] == "abc123"


def test_ollama_generation_only_provider_rejects_embedding_calls() -> None:
    client = httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(500, request=request)
        ),
        base_url="http://test",
    )
    provider = OllamaProvider("qwen", client=client)

    with pytest.raises(ValueError, match="embedding model"):
        provider.embed(["one"])


def test_ollama_embedding_adapter_uses_bounded_batches() -> None:
    batch_sizes: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        batch_sizes.append(len(body["input"]))
        return httpx.Response(
            200,
            json={"embeddings": [[float(len(text))] for text in body["input"]]},
        )

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://test")
    provider = OllamaProvider(
        "qwen", "embed", "http://test", client=client, embedding_batch_size=4
    )
    texts = [str(index) for index in range(9)]

    assert provider.embed(texts) == [[1.0]] * 9
    assert batch_sizes == [4, 4, 1]


def test_fake_provider_exhaustion_is_explicit() -> None:
    provider = FakeProvider(outputs=[])
    with pytest.raises(RuntimeError, match="exhausted"):
        provider.generate(GenerationRequest("test", "prompt", SCHEMA))


def test_result_metadata_never_contains_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "super-secret")
    provider = FakeProvider(outputs=[{"answer": "yes"}])
    result = provider.generate(GenerationRequest("test", "prompt", SCHEMA))
    assert "super-secret" not in result.model_dump_json()


def test_openai_provider_loads_project_dotenv_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded: list[tuple[object, bool]] = []
    monkeypatch.setattr(
        "vc_clone_graph.providers.openai.load_dotenv",
        lambda path, override: loaded.append((path, override)),
    )

    OpenAIProvider("test-model", client=object())  # type: ignore[arg-type]

    assert loaded == [(Path(".env"), False)]


def test_openai_provider_honors_configured_output_limit() -> None:
    captured: dict = {}

    class Responses:
        def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                output_text='{"answer":"yes"}',
                usage=SimpleNamespace(input_tokens=3, output_tokens=2),
                id="response-1",
            )

    client = SimpleNamespace(responses=Responses())
    provider = OpenAIProvider(
        "test-model", client=client, max_output_tokens=777  # type: ignore[arg-type]
    )

    provider.generate(GenerationRequest("test", "prompt", SCHEMA))

    assert captured["max_output_tokens"] == 777


def test_openai_provider_honors_smaller_request_output_limit() -> None:
    captured: dict = {}

    class Responses:
        def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                output_text='{"answer":"yes"}',
                usage=SimpleNamespace(input_tokens=3, output_tokens=2),
                id="response-1",
            )

    provider = OpenAIProvider(
        "test-model",
        client=SimpleNamespace(responses=Responses()),  # type: ignore[arg-type]
        max_output_tokens=8192,
    )
    provider.generate(
        GenerationRequest("phase1_plan", "prompt", SCHEMA, max_output_tokens=768)
    )
    assert captured["max_output_tokens"] == 768


def test_composite_provider_delegates_generation_and_embedding_separately() -> None:
    class Generator:
        def generate(self, request: GenerationRequest) -> GenerationResult:
            return GenerationResult(
                parsed={"answer": request.phase},
                content='{"answer":"phase1"}',
                usage=Usage(input_tokens=1),
                elapsed_seconds=0,
                raw_metadata={},
            )

    class Embedder:
        def embed(self, texts: list[str]) -> list[list[float]]:
            return [[float(len(text))] for text in texts]

    provider = CompositeProvider(Generator(), Embedder())

    assert provider.generate(GenerationRequest("phase1", "prompt", SCHEMA)).parsed == {
        "answer": "phase1"
    }
    assert provider.embed(["one", "three"]) == [[3.0], [5.0]]


def test_openrouter_provider_enforces_schema_routing_and_records_cost() -> None:
    captured: dict = {}

    class Completions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                id="generation-1",
                model="openai/gpt-5.6-luna",
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content='{"answer":"yes"}')
                    )
                ],
                usage=SimpleNamespace(
                    prompt_tokens=12,
                    completion_tokens=4,
                    prompt_tokens_details=SimpleNamespace(cached_tokens=3),
                    cost=0.00125,
                ),
            )

    client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
    provider = OpenRouterProvider(
        "openai/gpt-5.6-luna",
        client=client,  # type: ignore[arg-type]
        max_output_tokens=777,
        require_parameters=True,
        data_collection="deny",
    )

    result = provider.generate(
        GenerationRequest(
            "phase1", "prompt", SCHEMA, reasoning_effort="high"
        )
    )

    assert result.parsed == {"answer": "yes"}
    assert captured["model"] == "openai/gpt-5.6-luna"
    assert captured["max_tokens"] == 777
    assert captured["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "phase1_output",
            "strict": True,
            "schema": SCHEMA,
        },
    }
    assert captured["extra_body"] == {
        "provider": {"require_parameters": True, "data_collection": "deny"},
        "reasoning": {"effort": "high"},
    }
    assert result.usage.input_tokens == 12
    assert result.usage.cached_input_tokens == 3
    assert result.usage.output_tokens == 4
    assert result.usage.cost_usd == 0.00125
    assert result.raw_metadata["response_id"] == "generation-1"
    assert result.raw_metadata["requested_model"] == "openai/gpt-5.6-luna"
    assert result.raw_metadata["returned_model"] == "openai/gpt-5.6-luna"
    assert result.raw_metadata["reasoning_effort"] == "high"
    assert result.raw_metadata["physical_attempts"] == 1


def test_generation_request_rejects_unknown_reasoning_effort() -> None:
    with pytest.raises(ValueError, match="reasoning_effort"):
        GenerationRequest(  # type: ignore[arg-type]
            "phase1", "prompt", SCHEMA, reasoning_effort="extreme"
        )


def _openrouter_response() -> SimpleNamespace:
    return SimpleNamespace(
        id="generation-retried",
        model="returned-model",
        choices=[SimpleNamespace(message=SimpleNamespace(content='{"answer":"yes"}'))],
        usage=SimpleNamespace(cost=0.002),
    )


@pytest.mark.parametrize("status_code", [408, 409, 429, 500, 503])
def test_openrouter_retries_retryable_http_statuses_at_most_three_times(
    status_code: int,
) -> None:
    attempts = 0

    class Completions:
        def create(self, **kwargs):
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                request = httpx.Request("POST", "https://openrouter.test/chat")
                response = httpx.Response(status_code, request=request)
                raise APIStatusError("retry", response=response, body=None)
            return _openrouter_response()

    provider = OpenRouterProvider(
        "requested-model",
        client=SimpleNamespace(chat=SimpleNamespace(completions=Completions())),  # type: ignore[arg-type]
        sleeper=lambda delay: None,
    )

    result = provider.generate(
        GenerationRequest("phase1", "prompt", SCHEMA, reasoning_effort="high")
    )

    assert attempts == 3
    assert result.raw_metadata["physical_attempts"] == 3


@pytest.mark.parametrize("exception_type", [APIConnectionError, APITimeoutError])
def test_openrouter_retries_connection_and_timeout_errors(exception_type: type) -> None:
    attempts = 0
    request = httpx.Request("POST", "https://openrouter.test/chat")

    class Completions:
        def create(self, **kwargs):
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                if exception_type is APITimeoutError:
                    raise APITimeoutError(request)
                raise APIConnectionError(request=request)
            return _openrouter_response()

    provider = OpenRouterProvider(
        "requested-model",
        client=SimpleNamespace(chat=SimpleNamespace(completions=Completions())),  # type: ignore[arg-type]
        sleeper=lambda delay: None,
    )

    assert provider.generate(
        GenerationRequest("phase1", "prompt", SCHEMA)
    ).raw_metadata["physical_attempts"] == 3
    assert attempts == 3


@pytest.mark.parametrize("status_code", [400, 401, 403, 404, 422])
def test_openrouter_does_not_retry_other_http_4xx(status_code: int) -> None:
    attempts = 0

    class Completions:
        def create(self, **kwargs):
            nonlocal attempts
            attempts += 1
            request = httpx.Request("POST", "https://openrouter.test/chat")
            response = httpx.Response(status_code, request=request)
            raise APIStatusError("do not retry", response=response, body=None)

    provider = OpenRouterProvider(
        "requested-model",
        client=SimpleNamespace(chat=SimpleNamespace(completions=Completions())),  # type: ignore[arg-type]
    )

    with pytest.raises(APIStatusError):
        provider.generate(GenerationRequest("phase1", "prompt", SCHEMA))
    assert attempts == 1


def test_openrouter_does_not_retry_response_schema_errors() -> None:
    attempts = 0

    class Completions:
        def create(self, **kwargs):
            nonlocal attempts
            attempts += 1
            request = httpx.Request("POST", "https://openrouter.test/chat")
            response = httpx.Response(200, request=request)
            raise APIResponseValidationError(response, body={"invalid": True})

    provider = OpenRouterProvider(
        "requested-model",
        client=SimpleNamespace(chat=SimpleNamespace(completions=Completions())),  # type: ignore[arg-type]
    )

    with pytest.raises(APIResponseValidationError):
        provider.generate(GenerationRequest("phase1", "prompt", SCHEMA))
    assert attempts == 1


def test_openrouter_reports_error_metadata_when_response_has_no_choices() -> None:
    class Completions:
        def create(self, **kwargs):
            return SimpleNamespace(
                choices=None,
                model_extra={
                    "error": {"message": "upstream rejected the structured request"}
                },
            )

    provider = OpenRouterProvider(
        "requested-model",
        client=SimpleNamespace(chat=SimpleNamespace(completions=Completions())),  # type: ignore[arg-type]
    )

    with pytest.raises(
        ValueError,
        match="OpenRouter response contains no choices.*upstream rejected",
    ):
        provider.generate(GenerationRequest("phase2", "prompt", SCHEMA))


def test_openrouter_retries_no_choice_response_with_retryable_error_code() -> None:
    attempts = 0

    class Completions:
        def create(self, **kwargs):
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                return SimpleNamespace(
                    choices=None,
                    model_extra={
                        "error": {
                            "code": 429,
                            "message": "model is temporarily rate-limited upstream",
                        }
                    },
                )
            return _openrouter_response()

    provider = OpenRouterProvider(
        "requested-model",
        client=SimpleNamespace(chat=SimpleNamespace(completions=Completions())),  # type: ignore[arg-type]
        sleeper=lambda delay: None,
        random_fn=lambda: 0.0,
    )

    result = provider.generate(GenerationRequest("phase2", "prompt", SCHEMA))

    assert attempts == 3
    assert result.raw_metadata["physical_attempts"] == 3


def test_openrouter_retries_blank_structured_response_before_succeeding() -> None:
    attempts = 0
    response_formats: list[dict] = []
    prompts: list[str] = []
    extra_bodies: list[dict] = []

    class Completions:
        def create(self, **kwargs):
            nonlocal attempts
            attempts += 1
            response_formats.append(kwargs["response_format"])
            prompts.append(kwargs["messages"][0]["content"])
            extra_bodies.append(kwargs["extra_body"])
            if attempts == 1:
                return SimpleNamespace(
                    id=f"blank-{attempts}",
                    model="returned-model",
                    choices=[
                        SimpleNamespace(message=SimpleNamespace(content="  \n"))
                    ],
                    usage=SimpleNamespace(),
                )
            return _openrouter_response()

    provider = OpenRouterProvider(
        "requested-model",
        client=SimpleNamespace(chat=SimpleNamespace(completions=Completions())),  # type: ignore[arg-type]
        sleeper=lambda delay: None,
        random_fn=lambda: 0.0,
    )

    result = provider.generate(
        GenerationRequest("phase1", "prompt", SCHEMA, reasoning_effort="high")
    )

    assert attempts == 2
    assert result.parsed == {"answer": "yes"}
    assert result.raw_metadata["physical_attempts"] == 2
    assert response_formats[0]["type"] == "json_schema"
    assert response_formats[1] == {"type": "json_object"}
    assert prompts[0] == "prompt"
    assert "JSON Schema compatibility fallback" in prompts[1]
    assert json.dumps(SCHEMA, separators=(",", ":")) in prompts[1]
    assert extra_bodies[0]["reasoning"] == {"effort": "high"}
    assert extra_bodies[1]["reasoning"] == {"effort": "low"}
    assert result.raw_metadata["structured_output_mode"] == "json_object_fallback"


def test_openrouter_rejects_blank_structured_response_after_three_attempts() -> None:
    attempts = 0

    class Completions:
        def create(self, **kwargs):
            nonlocal attempts
            attempts += 1
            return SimpleNamespace(
                id=f"blank-{attempts}",
                model="returned-model",
                choices=[SimpleNamespace(message=SimpleNamespace(content=""))],
                usage=SimpleNamespace(),
            )

    provider = OpenRouterProvider(
        "requested-model",
        client=SimpleNamespace(chat=SimpleNamespace(completions=Completions())),  # type: ignore[arg-type]
        sleeper=lambda delay: None,
        random_fn=lambda: 0.0,
    )

    with pytest.raises(
        ValueError,
        match="blank structured content after 3 attempts.*blank-3",
    ):
        provider.generate(GenerationRequest("phase1", "prompt", SCHEMA))

    assert attempts == 3


def test_openrouter_rejects_injected_client_with_hidden_retries() -> None:
    client = SimpleNamespace(
        max_retries=2,
        chat=SimpleNamespace(completions=SimpleNamespace()),
    )

    with pytest.raises(ValueError, match="retries must be disabled"):
        OpenRouterProvider("model", client=client)  # type: ignore[arg-type]


def test_openrouter_accepts_injected_client_with_retries_disabled() -> None:
    client = SimpleNamespace(
        max_retries=0,
        chat=SimpleNamespace(completions=SimpleNamespace()),
    )

    provider = OpenRouterProvider("model", client=client)  # type: ignore[arg-type]

    assert provider.client is client


def test_openrouter_uses_bounded_exponential_backoff_and_stops_after_three() -> None:
    attempts = 0
    delays: list[float] = []

    class Completions:
        def create(self, **kwargs):
            nonlocal attempts
            attempts += 1
            request = httpx.Request("POST", "https://openrouter.test/chat")
            response = httpx.Response(503, request=request)
            raise APIStatusError("retry", response=response, body=None)

    provider = OpenRouterProvider(
        "model",
        client=SimpleNamespace(chat=SimpleNamespace(completions=Completions())),  # type: ignore[arg-type]
        sleeper=delays.append,
        random_fn=lambda: 0.0,
    )

    with pytest.raises(APIStatusError):
        provider.generate(GenerationRequest("phase1", "prompt", SCHEMA))

    assert attempts == 3
    assert delays == [0.25, 0.5]


@pytest.mark.parametrize(
    ("retry_after", "expected_delay"), [("2.5", 2.5), ("60", 5.0)]
)
def test_openrouter_honors_and_caps_retry_after_seconds(
    retry_after: str, expected_delay: float
) -> None:
    attempts = 0
    delays: list[float] = []

    class Completions:
        def create(self, **kwargs):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                request = httpx.Request("POST", "https://openrouter.test/chat")
                response = httpx.Response(
                    429, request=request, headers={"Retry-After": retry_after}
                )
                raise APIStatusError("retry", response=response, body=None)
            return _openrouter_response()

    provider = OpenRouterProvider(
        "model",
        client=SimpleNamespace(chat=SimpleNamespace(completions=Completions())),  # type: ignore[arg-type]
        sleeper=delays.append,
        random_fn=lambda: 0.0,
    )

    result = provider.generate(GenerationRequest("phase1", "prompt", SCHEMA))

    assert result.raw_metadata["physical_attempts"] == 2
    assert delays == [expected_delay]


def test_openrouter_connection_retry_uses_backoff() -> None:
    attempts = 0
    delays: list[float] = []
    request = httpx.Request("POST", "https://openrouter.test/chat")

    class Completions:
        def create(self, **kwargs):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise APIConnectionError(request=request)
            return _openrouter_response()

    provider = OpenRouterProvider(
        "model",
        client=SimpleNamespace(chat=SimpleNamespace(completions=Completions())),  # type: ignore[arg-type]
        sleeper=delays.append,
        random_fn=lambda: 0.0,
    )

    provider.generate(GenerationRequest("phase1", "prompt", SCHEMA))

    assert delays == [0.25]


def test_openrouter_request_cannot_raise_provider_output_limit() -> None:
    captured: dict = {}

    class Completions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content='{"answer":"yes"}'))],
                usage=SimpleNamespace(),
            )

    provider = OpenRouterProvider(
        "model",
        client=SimpleNamespace(  # type: ignore[arg-type]
            chat=SimpleNamespace(completions=Completions())
        ),
        max_output_tokens=8192,
    )
    provider.generate(
        GenerationRequest("phase1_plan", "prompt", SCHEMA, max_output_tokens=20_000)
    )
    assert captured["max_tokens"] == 8192


def test_ollama_provider_honors_smaller_request_output_limit() -> None:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(200, json={"message": {"content": '{"answer":"yes"}'}})

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://test")
    provider = OllamaProvider(
        "qwen", "embed", "http://test", client=client, max_output_tokens=8192
    )
    provider.generate(
        GenerationRequest("phase1_plan", "prompt", SCHEMA, max_output_tokens=768)
    )
    assert captured["options"]["num_predict"] == 768


def test_openrouter_provider_loads_key_without_serializing_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-secret")
    captured: dict = {}

    class Client:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr("vc_clone_graph.providers.openrouter.OpenAI", Client)
    provider = OpenRouterProvider("openai/gpt-5.6-luna")

    assert captured["api_key"] == "openrouter-secret"
    assert captured["max_retries"] == 0
    assert "openrouter-secret" not in repr(provider)


def test_openrouter_provider_configures_a_bounded_request_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-secret")
    captured: dict = {}

    class Client:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr("vc_clone_graph.providers.openrouter.OpenAI", Client)
    OpenRouterProvider("openai/gpt-5.6-luna", request_timeout_seconds=300)

    assert captured["timeout"] == 300
