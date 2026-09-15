"""Deterministic episode-grouped folds for personalized evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from .personalized_cases import PersonalizedCase


@dataclass(frozen=True)
class EpisodeFold:
    held_episode: str
    train_indices: tuple[int, ...]
    test_indices: tuple[int, ...]
    inner_splits: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]


def _inner_grouped_splits(
    cases: Sequence[PersonalizedCase],
    train_indices: tuple[int, ...],
    requested: int,
    seed: int,
) -> tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]:
    targets = np.asarray([cases[index].target for index in train_indices], dtype=int)
    groups = np.asarray([cases[index].group for index in train_indices], dtype=object)
    if set(targets.tolist()) != {0, 1}:
        raise ValueError("outer training fold must retain both decisions")
    maximum = min(
        requested,
        len(set(groups.tolist())),
        int(targets.sum()),
        int(len(targets) - targets.sum()),
    )
    for split_count in range(maximum, 1, -1):
        splitter = StratifiedGroupKFold(
            n_splits=split_count, shuffle=True, random_state=seed
        )
        candidate = []
        valid = True
        for local_train, local_valid in splitter.split(
            np.zeros((len(train_indices), 1)), targets, groups
        ):
            global_train = tuple(train_indices[int(index)] for index in local_train)
            global_valid = tuple(train_indices[int(index)] for index in local_valid)
            train_groups = {cases[index].group for index in global_train}
            valid_groups = {cases[index].group for index in global_valid}
            if (
                train_groups & valid_groups
                or {cases[index].target for index in global_train} != {0, 1}
            ):
                valid = False
                break
            candidate.append((global_train, global_valid))
        if valid:
            return tuple(candidate)
    raise ValueError("unable to create grouped inner folds with both decisions in training")


def make_episode_folds(
    cases: Sequence[PersonalizedCase], *, inner_splits: int, seed: int
) -> tuple[EpisodeFold, ...]:
    """Leave out every row from one episode and group inner validation likewise."""
    if not cases or inner_splits < 2:
        raise ValueError("episode folds require cases and at least two inner splits")
    if any(case.group != case.episode_slug for case in cases):
        raise ValueError("personalized outer group must equal episode_slug")
    episodes = sorted({case.episode_slug for case in cases})
    folds = []
    for fold_index, episode in enumerate(episodes):
        held = tuple(
            index for index, case in enumerate(cases) if case.episode_slug == episode
        )
        train = tuple(
            index for index, case in enumerate(cases) if case.episode_slug != episode
        )
        folds.append(EpisodeFold(
            held_episode=episode,
            train_indices=train,
            test_indices=held,
            inner_splits=_inner_grouped_splits(
                cases, train, inner_splits, seed + fold_index
            ),
        ))
    return tuple(folds)


def _manifest_payload(
    cases: Sequence[PersonalizedCase],
    folds: Sequence[EpisodeFold],
    source_registry_sha256: str,
) -> dict[str, object]:
    return {
        "schema": "personalized-episode-fold-manifest-v1",
        "source_registry_sha256": source_registry_sha256,
        "case_count": len(cases),
        "episode_count": len({case.episode_slug for case in cases}),
        "vc_count": len({case.vc_slug for case in cases}),
        "in_count": sum(case.target for case in cases),
        "cases": [
            {
                "index": index,
                "vc_slug": case.vc_slug,
                "episode_slug": case.episode_slug,
                "source_hashes": dict(sorted(case.source_hashes.items())),
            }
            for index, case in enumerate(cases)
        ],
        "folds": [
            {
                "held_episode": fold.held_episode,
                "train_indices": list(fold.train_indices),
                "test_indices": list(fold.test_indices),
                "excluded_rows": [
                    {
                        "vc_slug": cases[index].vc_slug,
                        "episode_slug": cases[index].episode_slug,
                    }
                    for index in fold.test_indices
                ],
                "inner_splits": [
                    {
                        "train_indices": list(train),
                        "valid_indices": list(valid),
                        "valid_episodes": sorted({cases[index].episode_slug for index in valid}),
                    }
                    for train, valid in fold.inner_splits
                ],
            }
            for fold in folds
        ],
    }


def write_fold_manifest(
    path: Path,
    cases: Sequence[PersonalizedCase],
    folds: Sequence[EpisodeFold],
    *,
    source_registry_sha256: str,
) -> str:
    """Write canonical JSON and its digest, returning the digest."""
    if len(source_registry_sha256) != 64:
        raise ValueError("source registry hash must be SHA-256")
    payload = _manifest_payload(cases, folds, source_registry_sha256)
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    digest = sha256(serialized.encode("utf-8")).hexdigest()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(serialized, encoding="utf-8")
    destination.with_suffix(".sha256").write_text(digest + "\n", encoding="utf-8")
    return digest
