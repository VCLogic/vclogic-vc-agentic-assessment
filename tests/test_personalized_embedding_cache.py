from __future__ import annotations

import numpy as np

from vc_clone_graph.personalized_embedding_cache import EmbeddingCache
from vc_clone_graph.personalized_features import chunk_text, embed_text_view


class StableProvider:
    def __init__(self) -> None:
        self.calls = 0
        self.metadata = {
            "backend": "test",
            "model": "stable",
            "revision": "abc123",
            "normalize": True,
            "document_prefix": "doc: ",
        }

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        return [
            [float(len(text)), float("late-signal" in text), float(index + 1)]
            for index, text in enumerate(texts)
        ]


def test_chunking_is_stable_gapless_and_preserves_late_content() -> None:
    text = ("First paragraph has evidence.\n\n" * 8) + "late-signal closes the pitch."

    first = chunk_text(text, max_chars=90, overlap_chars=15)
    second = chunk_text(text, max_chars=90, overlap_chars=15)

    assert first == second
    assert first[0].start == 0
    assert first[-1].end == len(text)
    assert all(left.end >= right.start for left, right in zip(first, first[1:]))
    assert "late-signal" in first[-1].text
    assert all(chunk.text == text[chunk.start:chunk.end] for chunk in first)


def test_embedding_cache_binds_source_model_and_chunk_policy(tmp_path) -> None:
    provider = StableProvider()
    cache = EmbeddingCache(tmp_path)
    text = ("traction " * 40) + "late-signal"

    first = embed_text_view(
        text,
        source_sha256="a" * 64,
        view="pitch",
        provider=provider,
        cache=cache,
        max_chars=80,
        overlap_chars=10,
    )
    second = embed_text_view(
        text,
        source_sha256="a" * 64,
        view="pitch",
        provider=provider,
        cache=cache,
        max_chars=80,
        overlap_chars=10,
    )

    assert provider.calls == 1
    assert first.cache_keys == second.cache_keys
    assert np.array_equal(first.pooled_mean, second.pooled_mean)
    assert first.pooled_max[1] == 1.0
    assert (tmp_path / "index.json").is_file()

    embed_text_view(
        text,
        source_sha256="b" * 64,
        view="pitch",
        provider=provider,
        cache=cache,
        max_chars=80,
        overlap_chars=10,
    )
    assert provider.calls == 2
