"""OpenRouter structured Chat Completions adapter."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from random import random
from time import perf_counter, sleep
from typing import Callable, Literal

from dotenv import load_dotenv
from openai import APIConnectionError, APIStatusError, OpenAI

from .base import GenerationRequest, GenerationResult, Usage


def _attribute(value: object | None, name: str, default: object = 0) -> object:
    if value is None:
        return default
    direct = getattr(value, name, None)
    if direct is not None:
        return direct
    extra = getattr(value, "model_extra", None)
    if isinstance(extra, dict) and extra.get(name) is not None:
        return extra[name]
    return default


class OpenRouterProvider:
    def __init__(
        self,
        model: str,
        *,
        base_url: str = "https://openrouter.ai/api/v1",
        api_key_env: str = "OPENROUTER_API_KEY",
        env_file: Path | None = Path(".env"),
        client: OpenAI | None = None,
        max_output_tokens: int = 4096,
        require_parameters: bool = True,
        data_collection: Literal["allow", "deny"] = "deny",
        request_timeout_seconds: int = 300,
        sleeper: Callable[[float], None] = sleep,
        random_fn: Callable[[], float] = random,
    ) -> None:
        if env_file is not None:
            load_dotenv(env_file, override=False)
        if client is None:
            api_key = os.environ.get(api_key_env)
            if not api_key:
                raise ValueError(f"missing required credential: {api_key_env}")
            client = OpenAI(
                api_key=api_key,
                base_url=base_url,
                max_retries=0,
                timeout=request_timeout_seconds,
            )
        elif getattr(client, "max_retries", 0) != 0:
            raise ValueError("injected OpenRouter client retries must be disabled")
        self.model = model
        self.client = client
        self.max_output_tokens = max_output_tokens
        self.require_parameters = require_parameters
        self.data_collection = data_collection
        self.sleeper = sleeper
        self.random_fn = random_fn

    @staticmethod
    def _retryable(exc: Exception) -> bool:
        if isinstance(exc, APIConnectionError):
            return True
        return isinstance(exc, APIStatusError) and (
            exc.status_code in {408, 409, 429}
            or 500 <= exc.status_code < 600
        )

    def _retry_delay(self, exc: Exception, physical_attempts: int) -> float:
        if isinstance(exc, APIStatusError):
            retry_after = exc.response.headers.get("Retry-After")
            if retry_after is not None:
                try:
                    seconds = float(retry_after)
                except ValueError:
                    pass
                else:
                    if math.isfinite(seconds):
                        return min(max(seconds, 0.0), 5.0)
        jitter = min(max(float(self.random_fn()), 0.0), 1.0) * 0.1
        return min(0.25 * (2 ** (physical_attempts - 1)) + jitter, 5.0)

    @staticmethod
    def _retryable_status_code(value: object) -> bool:
        try:
            status_code = int(value)
        except (TypeError, ValueError):
            return False
        return status_code in {408, 409, 429} or 500 <= status_code < 600

    def _default_retry_delay(self, physical_attempts: int) -> float:
        jitter = min(max(float(self.random_fn()), 0.0), 1.0) * 0.1
        return min(0.25 * (2 ** (physical_attempts - 1)) + jitter, 5.0)

    def generate(self, request: GenerationRequest) -> GenerationResult:
        started = perf_counter()
        limit = min(
            self.max_output_tokens,
            request.max_output_tokens or self.max_output_tokens,
        )
        extra_body: dict[str, object] = {
            "provider": {
                "require_parameters": self.require_parameters,
                "data_collection": self.data_collection,
            }
        }
        if request.reasoning_effort is not None:
            extra_body["reasoning"] = {"effort": request.reasoning_effort}
        physical_attempts = 0
        schema_fallback = False
        while True:
            physical_attempts += 1
            if schema_fallback:
                prompt = (
                    f"{request.prompt}\n\n"
                    "JSON Schema compatibility fallback: return one JSON object that "
                    "validates against this exact schema. Emit no markdown or commentary.\n"
                    f"{json.dumps(request.schema, separators=(',', ':'), ensure_ascii=False)}"
                )
                response_format = {"type": "json_object"}
            else:
                prompt = request.prompt
                response_format = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": f"{request.phase}_output",
                        "strict": True,
                        "schema": request.schema,
                    },
                }
            attempt_extra_body = dict(extra_body)
            if schema_fallback and request.reasoning_effort is not None:
                attempt_extra_body["reasoning"] = {"effort": "low"}
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=limit,
                    response_format=response_format,
                    extra_body=attempt_extra_body,
                )
            except Exception as exc:
                if physical_attempts >= 3 or not self._retryable(exc):
                    raise
                self.sleeper(self._retry_delay(exc, physical_attempts))
                continue

            choices = getattr(response, "choices", None)
            if choices:
                candidate_content = getattr(choices[0].message, "content", None)
                if isinstance(candidate_content, str) and candidate_content.strip():
                    break
                if physical_attempts < 3:
                    schema_fallback = True
                    self.sleeper(self._default_retry_delay(physical_attempts))
                    continue
                response_id = getattr(response, "id", None)
                finish_reason = getattr(choices[0], "finish_reason", None)
                raise ValueError(
                    "OpenRouter response contains blank structured content after "
                    f"{physical_attempts} attempts (response_id={response_id}, "
                    f"finish_reason={finish_reason}, fallback=json_object)"
                )
            error = _attribute(response, "error", None)
            error_code = error.get("code") if isinstance(error, dict) else None
            if (
                physical_attempts < 3
                and self._retryable_status_code(error_code)
            ):
                self.sleeper(self._default_retry_delay(physical_attempts))
                continue
            details = (
                json.dumps(error, sort_keys=True, default=str)
                if error
                else "no error metadata"
            )
            raise ValueError(
                f"OpenRouter response contains no choices: {details}"
            )
        content = choices[0].message.content
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            parsed = None
        native_usage = getattr(response, "usage", None)
        prompt_details = getattr(native_usage, "prompt_tokens_details", None)
        cost = float(_attribute(native_usage, "cost", 0.0))
        return GenerationResult(
            parsed=parsed,
            content=content,
            usage=Usage(
                input_tokens=int(_attribute(native_usage, "prompt_tokens")),
                cached_input_tokens=int(_attribute(prompt_details, "cached_tokens")),
                output_tokens=int(_attribute(native_usage, "completion_tokens")),
                cost_usd=cost,
            ),
            elapsed_seconds=perf_counter() - started,
            raw_metadata={
                "provider": "openrouter",
                "requested_model": self.model,
                "returned_model": getattr(response, "model", self.model),
                "reasoning_effort": request.reasoning_effort,
                "response_id": getattr(response, "id", None),
                "cost_usd": cost,
                "physical_attempts": physical_attempts,
                "structured_output_mode": (
                    "json_object_fallback" if schema_fallback else "json_schema"
                ),
                # Preserve the original adapter metadata keys for old artifacts.
                "model": getattr(response, "model", self.model),
                "id": getattr(response, "id", None),
            },
        )
