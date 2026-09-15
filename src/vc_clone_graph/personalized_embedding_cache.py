"""Source- and model-bound local cache for frozen text embeddings."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Mapping

import numpy as np


class EmbeddingCache:
    """Persist embedding matrices under canonical metadata hashes."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / "index.json"
        if self.index_path.exists():
            payload = json.loads(self.index_path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("embedding cache index must be an object")
            self._index: dict[str, dict[str, object]] = payload
        else:
            self._index = {}

    @staticmethod
    def key(metadata: Mapping[str, object]) -> str:
        serialized = json.dumps(metadata, sort_keys=True, separators=(",", ":"))
        return sha256(serialized.encode("utf-8")).hexdigest()

    def load(self, key: str) -> np.ndarray | None:
        row = self._index.get(key)
        if row is None:
            return None
        path = self.root / str(row["file"])
        if not path.is_file():
            raise ValueError(f"embedding cache file is missing: {path}")
        with path.open("rb") as stream:
            matrix = np.load(stream, allow_pickle=False)
        if matrix.ndim != 2 or list(matrix.shape) != row.get("shape"):
            raise ValueError(f"embedding cache shape mismatch: {key}")
        return np.asarray(matrix, dtype=float)

    def store(
        self, key: str, matrix: np.ndarray, metadata: Mapping[str, object]
    ) -> None:
        values = np.asarray(matrix, dtype=float)
        if values.ndim != 2 or not values.size or not np.isfinite(values).all():
            raise ValueError("embedding cache requires a finite non-empty matrix")
        filename = f"{key}.npy"
        destination = self.root / filename
        temporary = self.root / f".{key}.tmp"
        with temporary.open("wb") as stream:
            np.save(stream, values, allow_pickle=False)
        temporary.replace(destination)
        self._index[key] = {
            "file": filename,
            "shape": list(values.shape),
            "metadata": dict(metadata),
        }
        index_temporary = self.root / ".index.tmp"
        index_temporary.write_text(
            json.dumps(self._index, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        index_temporary.replace(self.index_path)
