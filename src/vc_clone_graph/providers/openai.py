"""OpenAI Responses API adapter for the production provider path."""

from __future__ import annotations

import json
import os
from pathlib import Path
from time import perf_counter
from typing import Sequence

from dotenv import load_dotenv
from openai import OpenAI

from .base import GenerationRequest, GenerationResult, Usage


class OpenAIProvider:
    def __init__(
        self,
        model: str,
        embedding_model: str = "text-embedding-3-small",
        *,
        env_file: Path | None = Path(".env"),
        client: OpenAI | None = None,
        max_output_tokens: int = 4096,
    ) -> None:
        if env_file is not None:
            load_dotenv(env_file, override=False)
        self.model = model
        self.embedding_model = embedding_model
        self.max_output_tokens = max_output_tokens
        self.client = client or OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    def generate(self, request: GenerationRequest) -> GenerationResult:
        started = perf_counter()
        limit = min(
            self.max_output_tokens,
            request.max_output_tokens or self.max_output_tokens,
        )
        response = self.client.responses.create(
            model=self.model,
            input=request.prompt,
            max_output_tokens=limit,
            text={
                "format": {
                    "type": "json_schema",
                    "name": f"{request.phase}_output",
                    "schema": request.schema,
                    "strict": True,
                }
            },
        )
        content = response.output_text
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            parsed = None
        usage = response.usage
        return GenerationResult(
            parsed=parsed,
            content=content,
            usage=Usage(
                input_tokens=getattr(usage, "input_tokens", 0),
                cached_input_tokens=getattr(
                    getattr(usage, "input_tokens_details", None), "cached_tokens", 0
                ),
                output_tokens=getattr(usage, "output_tokens", 0),
            ),
            elapsed_seconds=perf_counter() - started,
            raw_metadata={"provider": "openai", "model": self.model, "id": response.id},
        )

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        response = self.client.embeddings.create(
            model=self.embedding_model, input=list(texts)
        )
        return [row.embedding for row in response.data]
