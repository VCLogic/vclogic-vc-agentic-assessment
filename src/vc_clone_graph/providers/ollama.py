"""Ollama chat and embedding adapter."""

from __future__ import annotations

import json
from time import perf_counter
from typing import Sequence

import httpx

from .base import GenerationRequest, GenerationResult, Usage


class OllamaProvider:
    def __init__(
        self,
        model: str,
        embedding_model: str | None = None,
        base_url: str = "http://127.0.0.1:11434",
        *,
        client: httpx.Client | None = None,
        timeout_seconds: float = 900,
        embedding_batch_size: int = 1,
        max_output_tokens: int = 4096,
        enable_thinking: bool = False,
        context_window: int = 32768,
        structured_output_mode: str = "schema",
    ) -> None:
        if embedding_batch_size < 1:
            raise ValueError("embedding_batch_size must be positive")
        self.model = model
        self.embedding_model = embedding_model
        self.embedding_batch_size = embedding_batch_size
        self.max_output_tokens = max_output_tokens
        self.enable_thinking = enable_thinking
        self.context_window = context_window
        if structured_output_mode not in {"schema", "json"}:
            raise ValueError("structured_output_mode must be schema or json")
        self.structured_output_mode = structured_output_mode
        self.client = client or httpx.Client(base_url=base_url, timeout=timeout_seconds)

    def generate(self, request: GenerationRequest) -> GenerationResult:
        started = perf_counter()
        limit = min(
            self.max_output_tokens,
            request.max_output_tokens or self.max_output_tokens,
        )
        prompt = request.prompt
        if self.structured_output_mode == "json":
            compact_schema = json.dumps(request.schema, separators=(",", ":"))
            prompt = (
                f"{prompt}\n\nReturn exactly one JSON object matching this schema:"
                f"\n{compact_schema}"
            )
        response = self.client.post(
            "/api/chat",
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "format": request.schema if self.structured_output_mode == "schema" else "json",
                "stream": False,
                "think": self.enable_thinking,
                "options": {
                    "temperature": 0.2,
                    "num_ctx": self.context_window,
                    "num_predict": limit,
                },
            },
        )
        response.raise_for_status()
        raw = response.json()
        message = raw.get("message", {})
        content = message.get("content", "")
        try:
            parsed = json.loads(content)
        except (TypeError, json.JSONDecodeError):
            parsed = None
        return GenerationResult(
            parsed=parsed,
            content=content if isinstance(content, str) else "",
            usage=Usage(
                input_tokens=int(raw.get("prompt_eval_count", 0)),
                output_tokens=int(raw.get("eval_count", 0)),
            ),
            elapsed_seconds=perf_counter() - started,
            raw_metadata={
                "provider": "ollama",
                "model": raw.get("model", self.model),
                "done_reason": raw.get("done_reason"),
                "thinking": message.get("thinking", ""),
            },
        )

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not self.embedding_model:
            raise ValueError("Ollama embedding model is not configured")
        vectors: list[list[float]] = []
        pending = list(texts)
        for start in range(0, len(pending), self.embedding_batch_size):
            batch = pending[start : start + self.embedding_batch_size]
            response = self.client.post(
                "/api/embed", json={"model": self.embedding_model, "input": batch}
            )
            response.raise_for_status()
            batch_vectors = response.json().get("embeddings")
            if type(batch_vectors) is not list or len(batch_vectors) != len(batch):
                raise ValueError("Ollama returned invalid embeddings")
            vectors.extend(batch_vectors)
        return vectors

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self.embed(texts)

    def embed_queries(self, texts: Sequence[str]) -> list[list[float]]:
        return self.embed(texts)

    @property
    def metadata(self) -> dict[str, object]:
        return {
            "backend": "ollama",
            "model": self.embedding_model,
            "revision": None,
            "normalize": False,
        }
