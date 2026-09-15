"""In-process, task-aware Sentence Transformers embedding adapter."""

from __future__ import annotations

from typing import Any, Sequence


class SentenceTransformerEmbeddingProvider:
    """Encode retrieval documents and queries with explicit task prefixes."""

    def __init__(
        self,
        model: str,
        *,
        revision: str,
        device: str = "auto",
        batch_size: int = 16,
        normalize: bool = True,
        document_prefix: str = "search_document: ",
        query_prefix: str = "search_query: ",
        model_instance: Any | None = None,
    ) -> None:
        if not revision:
            raise ValueError("Sentence Transformers model revision must be pinned")
        if batch_size < 1:
            raise ValueError("embedding batch size must be positive")
        self.model_name = model
        self.revision = revision
        self.device = device
        self.batch_size = batch_size
        self.normalize = normalize
        self.document_prefix = document_prefix
        self.query_prefix = query_prefix
        if model_instance is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:  # pragma: no cover - environment dependent
                raise RuntimeError(
                    "Sentence Transformers embeddings require: uv sync --extra embeddings"
                ) from exc
            model_instance = SentenceTransformer(
                model,
                revision=revision,
                device=None if device == "auto" else device,
                trust_remote_code=False,
            )
        self.model = model_instance

    def _encode(self, texts: Sequence[str], prefix: str) -> list[list[float]]:
        prepared = [f"{prefix}{text}" for text in texts]
        if not prepared:
            return []
        vectors = self.model.encode(
            prepared,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        raw = vectors.tolist() if hasattr(vectors, "tolist") else vectors
        return [[float(value) for value in vector] for vector in raw]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self._encode(texts, self.document_prefix)

    def embed_queries(self, texts: Sequence[str]) -> list[list[float]]:
        return self._encode(texts, self.query_prefix)

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Backward-compatible alias; unqualified text is treated as a document."""
        return self.embed_documents(texts)

    @property
    def metadata(self) -> dict[str, object]:
        return {
            "backend": "sentence_transformers",
            "model": self.model_name,
            "revision": self.revision,
            "normalize": self.normalize,
            "document_prefix": self.document_prefix,
            "query_prefix": self.query_prefix,
        }
